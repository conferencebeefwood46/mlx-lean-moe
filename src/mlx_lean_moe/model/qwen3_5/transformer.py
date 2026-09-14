"""The qwen3_5 decoder stack: embedding, 40 layers, norm, head.

Layers alternate between gated-delta linear attention and softmax attention
on ``full_attention_interval``. Routed experts stream per layer; everything
else is resident.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mlx.core as mx

from mlx_lean_moe.cache.kv_cache import GrowingKVCache
from mlx_lean_moe.cache.recurrent_cache import RecurrentCache
from mlx_lean_moe.model.moe_block import quantized_linear
from mlx_lean_moe.model.qwen3_5.attention import Qwen3_5Attention
from mlx_lean_moe.model.qwen3_5.config import Qwen3_5Config
from mlx_lean_moe.model.qwen3_5.linear_attention import Qwen3_5LinearAttention
from mlx_lean_moe.model.qwen3_5.mlp import (
    Qwen3_5Experts,
    Qwen3_5Router,
    Qwen3_5SharedExpert,
)
from mlx_lean_moe.weights import expert_pack
from mlx_lean_moe.weights.expert_loader import (
    CachedExpertLoader,
    TensorStreamer,
    read_linear,
    validate_linear_layout,
)
from mlx_lean_moe.weights.safetensors_index import build_index
from mlx_lean_moe.weights.stacked_expert_loader import StackedExpertStreamer


class Qwen3_5DecoderLayer:
    def __init__(
        self,
        config: Qwen3_5Config,
        layer_index: int,
        streamer: TensorStreamer,
        expert_loader: CachedExpertLoader,
    ) -> None:
        prefix = f"{config.tensor_prefix}.layers.{layer_index}"
        self.config = config
        self.eps = config.rms_norm_eps
        self.is_linear = config.is_linear_per_layer[layer_index]
        self.input_layernorm = streamer.read_tensor(f"{prefix}.input_layernorm.weight")
        self.post_attention_layernorm = streamer.read_tensor(
            f"{prefix}.post_attention_layernorm.weight"
        )

        if self.is_linear:
            attn_prefix = f"{prefix}.linear_attn"
            self.attention = Qwen3_5LinearAttention(
                config,
                in_proj_qkv=self._read_linear(streamer, f"{attn_prefix}.in_proj_qkv"),
                in_proj_z=self._read_linear(streamer, f"{attn_prefix}.in_proj_z"),
                in_proj_a=self._read_linear(streamer, f"{attn_prefix}.in_proj_a"),
                in_proj_b=self._read_linear(streamer, f"{attn_prefix}.in_proj_b"),
                conv1d_weight=streamer.read_tensor(f"{attn_prefix}.conv1d.weight"),
                A_log=streamer.read_tensor(f"{attn_prefix}.A_log"),
                dt_bias=streamer.read_tensor(f"{attn_prefix}.dt_bias"),
                norm_weight=streamer.read_tensor(f"{attn_prefix}.norm.weight"),
                out_proj=self._read_linear(streamer, f"{attn_prefix}.out_proj"),
            )
        else:
            attn_prefix = f"{prefix}.self_attn"
            self.attention = Qwen3_5Attention(
                config,
                q_proj=self._read_linear(streamer, f"{attn_prefix}.q_proj"),
                q_norm=streamer.read_tensor(f"{attn_prefix}.q_norm.weight"),
                k_proj=self._read_linear(streamer, f"{attn_prefix}.k_proj"),
                k_norm=streamer.read_tensor(f"{attn_prefix}.k_norm.weight"),
                v_proj=self._read_linear(streamer, f"{attn_prefix}.v_proj"),
                o_proj=self._read_linear(streamer, f"{attn_prefix}.o_proj"),
            )

        mlp_prefix = f"{prefix}.mlp"
        self.router = Qwen3_5Router(
            config,
            gate=self._read_linear(streamer, f"{mlp_prefix}.gate", config.router_quant),
        )
        self.experts = Qwen3_5Experts(config, layer_index, expert_loader)
        self.shared_expert = Qwen3_5SharedExpert(
            config,
            projections={
                proj: self._read_linear(streamer, f"{mlp_prefix}.shared_expert.{proj}")
                for proj in ("gate_proj", "up_proj", "down_proj")
            },
            gate=self._read_linear(
                streamer, f"{mlp_prefix}.shared_expert_gate", config.shared_gate_quant
            ),
        )

    def _read_linear(self, streamer: TensorStreamer, name: str, fallback=None):
        return read_linear(streamer, name, self.config.quant_for(name, fallback))

    def __call__(self, x: mx.array, cache) -> mx.array:
        """``x``: ``(hidden_size,)`` for a decode step, or
        ``(L, hidden_size)`` for a batched prefill."""
        if x.ndim not in (1, 2):
            raise ValueError(
                f"Qwen3_5DecoderLayer expects 1-D or 2-D x, got shape {x.shape}"
            )
        squeeze = x.ndim == 1
        rows = x[None, :] if squeeze else x

        h = rows + self.attention(
            mx.fast.rms_norm(rows, self.input_layernorm, self.eps), cache
        )

        normed = mx.fast.rms_norm(h, self.post_attention_layernorm, self.eps)
        indices, weights = self.router(normed)
        # Dispatched into the expert-read window, the one stretch where the
        # device is otherwise idle.
        shared = self.shared_expert(normed)
        routed = self.experts(
            normed, indices, weights, before_load=lambda: mx.async_eval(shared)
        )
        out = h + routed + shared
        return out[0] if squeeze else out


class Qwen3_5Transformer:
    """Owns all resident weights, the per-layer expert caches and the
    per-layer attention state for one generation session."""

    def __init__(
        self,
        model_dir: str | Path,
        config: Qwen3_5Config,
        max_context: int,
        expert_cache_size_per_layer: int | None = None,
        max_concurrent_expert_reads: int | None = None,
        prefill_chunk_size: int = 128,
    ) -> None:
        if max_context <= 0:
            raise ValueError("max_context must be positive")
        if prefill_chunk_size <= 0:
            raise ValueError("prefill_chunk_size must be positive")
        self.config = config
        self.prefill_chunk_size = prefill_chunk_size
        self.model_dir = Path(model_dir)
        self.max_context = max_context
        index = build_index(self.model_dir)
        self.streamer = TensorStreamer(self.model_dir, index)

        max_workers = max_concurrent_expert_reads or (config.experts_per_token * 9)
        self._read_executor = ThreadPoolExecutor(max_workers=max_workers)

        # A pack holds each expert's nine byte ranges as one. Absent, the
        # stacked tensors are read directly.
        layout = expert_pack.load_layout(self.model_dir)
        if layout is not None and layout.num_experts != config.num_experts:
            raise ValueError(
                f"expert pack holds {layout.num_experts} experts, checkpoint has {config.num_experts}"
            )
        self.stacked_streamer = (
            expert_pack.ExpertPackStreamer(self.model_dir, layout)
            if layout is not None
            else StackedExpertStreamer(
                self.model_dir,
                index,
                num_experts=config.num_experts,
                layer_prefix=f"{config.tensor_prefix}.layers",
                stack_name="mlp.switch_mlp",
            )
        )
        for layer_index in range(config.num_layers):
            prefix = f"{config.tensor_prefix}.layers.{layer_index}.mlp.switch_mlp"
            for projection in ("gate_proj", "up_proj", "down_proj"):
                name = f"{prefix}.{projection}"
                validate_linear_layout(
                    index, name, config.quant_for(name, config.expert_quant)
                )
        cache_size = expert_cache_size_per_layer or config.experts_per_token
        self.expert_loaders = [
            CachedExpertLoader(
                self.stacked_streamer,
                max_size=cache_size,
                executor=self._read_executor,
            )
            for _ in range(config.num_layers)
        ]

        # Named, not loaded: a token needs one row of the table. See
        # `_embed_rows`.
        self.embed_prefix = f"{config.tensor_prefix}.embed_tokens"
        self.embed_quant = config.quant_for(self.embed_prefix)
        self.embed_is_quantized = validate_linear_layout(
            index, self.embed_prefix, self.embed_quant
        )
        self.final_norm = self.streamer.read_tensor(
            f"{config.tensor_prefix}.norm.weight"
        )
        # A real tensor, embeddings not being tied here, and it sits beside
        # `model` rather than inside it.
        lm_head_prefix = config.tensor_prefix.rsplit(".", 1)[0] + ".lm_head"
        self.lm_head = read_linear(
            self.streamer, lm_head_prefix, config.quant_for(lm_head_prefix)
        )

        self.layers = [
            Qwen3_5DecoderLayer(config, i, self.streamer, self.expert_loaders[i])
            for i in range(config.num_layers)
        ]
        self.cache = self._new_cache()

    def _new_cache(self) -> list:
        config = self.config
        caches = []
        for is_linear in config.is_linear_per_layer:
            if is_linear:
                caches.append(
                    RecurrentCache(
                        conv_kernel_dim=config.linear.conv_kernel_dim,
                        conv_dim=config.linear.conv_dim,
                        num_value_heads=config.linear.num_value_heads,
                        value_head_dim=config.linear.value_head_dim,
                        key_head_dim=config.linear.key_head_dim,
                    )
                )
            else:
                caches.append(
                    GrowingKVCache(
                        self.max_context,
                        num_kv_heads=config.num_key_value_heads,
                        head_dim=config.head_dim,
                    )
                )
        return caches

    def reset_cache(self) -> None:
        """Clears conversation state in place, keeping resident weights and
        the expert loaders' warm caches (see `ChatSession.reset`)."""
        self.cache = self._new_cache()

    def _embed_rows(self, token_ids: list[int]) -> mx.array:
        """Reads and dequantizes only the rows these tokens need, rather
        than holding the table resident."""
        return self.streamer.read_linear_rows(
            self.embed_prefix,
            token_ids,
            self.embed_quant,
            quantized=self.embed_is_quantized,
        ).astype(mx.float32)

    def _lm_head(self, x: mx.array) -> mx.array:
        return quantized_linear(x[None, :], self.lm_head, self.config.other_quant)[0]

    def __call__(self, token_id: int) -> mx.array:
        """One decode step: token in, next-token logits out."""
        self._check_context(1)
        x = self._embed_rows([token_id])[0]
        for i, layer in enumerate(self.layers):
            x = layer(x, self.cache[i])
        x = mx.fast.rms_norm(x, self.final_norm, self.config.rms_norm_eps)
        return self._lm_head(x)

    def _check_context(self, count: int) -> None:
        if self.cache and self.cache[0].size + count > self.max_context:
            raise IndexError(
                f"Qwen3_5Transformer exceeded its capacity of {self.max_context} tokens"
            )

    def prefill(self, token_ids: list[int]) -> mx.array:
        """Process a prompt in bounded chunks, returning only final logits.

        Both cache kinds carry state across chunks, so context is preserved.
        """
        if not token_ids:
            raise ValueError("prefill requires at least one token")
        self._check_context(len(token_ids))
        for start in range(0, len(token_ids), self.prefill_chunk_size):
            x = self._embed_rows(token_ids[start : start + self.prefill_chunk_size])
            for i, layer in enumerate(self.layers):
                x = layer(x, self.cache[i])
                # The carried state too: a cache must not retain unevaluated
                # work from an earlier chunk.
                mx.eval(x, *self.cache[i].state())
        x = mx.fast.rms_norm(x[-1], self.final_norm, self.config.rms_norm_eps)
        return self._lm_head(x)

    def close(self) -> None:
        self._read_executor.shutdown(wait=True)
        self.stacked_streamer.close()
        self.streamer.close()
