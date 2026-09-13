"""qwen3_5's softmax attention: grouped-query, partial-rotary RoPE, and an
output gate fused into ``q_proj``'s second half rather than stored as its
own tensor."""

import mlx.core as mx

from mlx_lean_moe.cache.kv_cache import KVCache
from mlx_lean_moe.model.moe_block import LinearWeights, QuantizedTensor, quantized_linear
from mlx_lean_moe.model.qwen3_5.config import Qwen3_5Config
from mlx_lean_moe.model.rope import apply_leading_rope


class Qwen3_5Attention:
    def __init__(
        self,
        config: Qwen3_5Config,
        q_proj: LinearWeights | QuantizedTensor,
        q_norm: mx.array,
        k_proj: LinearWeights | QuantizedTensor,
        k_norm: mx.array,
        v_proj: LinearWeights | QuantizedTensor,
        o_proj: LinearWeights | QuantizedTensor,
    ) -> None:
        self.config = config
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = config.head_dim**-0.5
        self.q_proj = q_proj
        self.q_norm = q_norm
        self.k_proj = k_proj
        self.k_norm = k_norm
        self.v_proj = v_proj
        self.o_proj = o_proj

    def __call__(self, x: mx.array, cache: KVCache) -> mx.array:
        """``x``: ``(hidden_size,)`` for one decode step, or
        ``(L, hidden_size)`` for a batched prefill. Returns the same rank."""
        if x.ndim not in (1, 2):
            raise ValueError(f"Qwen3_5Attention expects 1-D or 2-D x, got shape {x.shape}")
        squeeze = x.ndim == 1
        rows = x[None, :] if squeeze else x
        seq_len = rows.shape[0]
        quant = self.config.other_quant
        eps = self.config.rms_norm_eps

        # q_proj carries the queries and the output gate side by side, per
        # head: (L, n_heads, 2 * head_dim) split down the middle.
        q_and_gate = quantized_linear(rows, self.q_proj, quant).reshape(seq_len, self.n_heads, 2 * self.head_dim)
        q, gate = mx.split(q_and_gate, 2, axis=-1)
        q = mx.fast.rms_norm(q, self.q_norm, eps)

        k = quantized_linear(rows, self.k_proj, quant).reshape(seq_len, self.n_kv_heads, self.head_dim)
        k = mx.fast.rms_norm(k, self.k_norm, eps)
        v = quantized_linear(rows, self.v_proj, quant).reshape(seq_len, self.n_kv_heads, self.head_dim)

        q = q[None].transpose(0, 2, 1, 3)  # (1, n_heads, L, head_dim)
        k = k[None].transpose(0, 2, 1, 3)
        v = v[None].transpose(0, 2, 1, 3)

        offset = cache.size
        # Leading-dims partial rotary, *not* the half-split pairing --
        # see model/rope.py, the two are genuinely different rotations.
        q = apply_leading_rope(q, self.config.rotary_dim, self.config.rope_theta, offset)
        k = apply_leading_rope(k, self.config.rotary_dim, self.config.rope_theta, offset)

        cache.append_many(k[0], v[0])
        keys, values = cache.state()  # (n_kv_heads, T_kv, head_dim), already head-major
        keys, values = keys[None], values[None]

        mask = "causal" if seq_len > 1 else None
        out = mx.fast.scaled_dot_product_attention(q, keys, values, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(seq_len, -1)  # (L, n_heads * head_dim)

        out = out * mx.sigmoid(gate.reshape(seq_len, -1))
        out = quantized_linear(out, self.o_proj, quant)
        return out[0] if squeeze else out
