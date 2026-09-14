"""The qwen3_5 config: shapes, per-layer kinds, and quantization schemes
resolved per projection name.
"""

from dataclasses import dataclass

from mlx_lean_moe.config import QuantScheme


@dataclass(frozen=True, slots=True)
class Qwen3_5LinearAttention:
    """Shape of the gated-delta layers. Key and value head counts differ, so
    q/k are repeated to match v before the recurrence."""

    num_key_heads: int
    num_value_heads: int
    key_head_dim: int
    value_head_dim: int
    conv_kernel_dim: int

    @property
    def key_dim(self) -> int:
        return self.num_key_heads * self.key_head_dim

    @property
    def value_dim(self) -> int:
        return self.num_value_heads * self.value_head_dim

    @property
    def conv_dim(self) -> int:
        """The convolution runs over q, k and v concatenated."""
        return self.key_dim * 2 + self.value_dim


@dataclass(frozen=True, slots=True)
class Qwen3_5Config:
    num_layers: int
    hidden_size: int
    vocab_size: int
    rms_norm_eps: float

    # Softmax-attention layers (the minority -- one in every
    # `full_attention_interval`).
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rope_theta: float
    rotary_dim: int  # head_dim * partial_rotary_factor

    # True where a layer uses gated-delta linear attention instead.
    is_linear_per_layer: tuple[bool, ...]
    linear: Qwen3_5LinearAttention

    # Routed experts, stored stacked per projection, plus one always-on
    # sigmoid-gated shared expert.
    num_experts: int
    experts_per_token: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    norm_topk_prob: bool

    expert_quant: QuantScheme
    router_quant: QuantScheme  # mlp.gate
    shared_gate_quant: QuantScheme  # mlp.shared_expert_gate
    other_quant: QuantScheme  # attention, shared expert, embeddings, lm_head

    tensor_prefix: str = "language_model.model"
    quant_overrides: tuple[tuple[str, QuantScheme], ...] = ()

    def quant_for(self, name: str, fallback: QuantScheme | None = None) -> QuantScheme:
        """Resolve one projection's exact mixed-precision override."""

        for tensor_name, scheme in self.quant_overrides:
            if tensor_name == name:
                return scheme

        return fallback or self.other_quant


def _quant_scheme(quant: dict, bits: int = 4, group_size: int = 64) -> QuantScheme:
    return QuantScheme(
        bits=int(quant.get("bits", bits)),
        group_size=int(quant.get("group_size", group_size)),
        mode=str(quant.get("mode", "affine")),
    )


def _quant_overrides(
    quant: dict, default: QuantScheme
) -> tuple[tuple[str, QuantScheme], ...]:
    overrides = []
    for name, value in quant.items():
        if isinstance(value, dict):
            overrides.append(
                (
                    name.removesuffix(".weight")
                    .removesuffix(".scales")
                    .removesuffix(".biases"),
                    QuantScheme(
                        bits=int(value.get("bits", default.bits)),
                        group_size=int(value.get("group_size", default.group_size)),
                        mode=str(value.get("mode", default.mode)),
                    ),
                )
            )

    return tuple(overrides)


def _quant_override(quant: dict, name_suffix: str, default: QuantScheme) -> QuantScheme:
    """Per-tensor-name overrides: every layer publishes its own identical
    entry, so the first match is representative."""

    for name, override in quant.items():
        if isinstance(override, dict) and name.endswith(name_suffix):
            return QuantScheme(
                bits=int(override.get("bits", default.bits)),
                group_size=int(override.get("group_size", default.group_size)),
                mode=str(override.get("mode", default.mode)),
            )

    return default


def _layer_types(text: dict, num_layers: int) -> list[str]:
    layer_types = text.get("layer_types")
    if layer_types is not None:
        return list(layer_types)

    # Fallback matching the reference implementation's own rule: a layer is
    # linear unless it's the last of each `full_attention_interval` group.
    interval = int(text.get("full_attention_interval", 4))

    return [
        "full_attention" if (i + 1) % interval == 0 else "linear_attention"
        for i in range(num_layers)
    ]


def from_hf_config(hf_config: dict) -> Qwen3_5Config:
    text = hf_config.get("text_config", hf_config)
    num_layers = int(text["num_hidden_layers"])

    # `mtp_num_hidden_layers` is deliberately not an error: the
    # multi-token-prediction head sits outside the `num_hidden_layers` stack.
    if text.get("mlp_only_layers"):
        raise ValueError(
            f"qwen3_5 config has mlp_only_layers={text['mlp_only_layers']!r}, but dense-only layers aren't implemented"
        )

    if int(text.get("decoder_sparse_step", 1)) != 1:
        raise ValueError(
            f"qwen3_5 config has decoder_sparse_step={text['decoder_sparse_step']!r}; only every-layer MoE (1) is implemented"
        )

    if not int(text.get("num_experts", 0)):
        raise ValueError(
            "qwen3_5 config has no experts -- this is a dense checkpoint, out of scope for this project"
        )

    layer_types = _layer_types(text, num_layers)
    is_linear_per_layer = tuple(lt == "linear_attention" for lt in layer_types)

    rope = text.get("rope_parameters") or hf_config.get("rope_parameters") or {}
    head_dim = int(
        text.get("head_dim")
        or (int(text["hidden_size"]) // int(text["num_attention_heads"]))
    )
    partial_rotary_factor = float(
        rope.get("partial_rotary_factor", hf_config.get("partial_rotary_factor", 1.0))
    )

    quant = hf_config.get("quantization") or hf_config.get("quantization_config") or {}
    default_quant = _quant_scheme(quant)

    return Qwen3_5Config(
        num_layers=num_layers,
        hidden_size=int(text["hidden_size"]),
        vocab_size=int(text["vocab_size"]),
        rms_norm_eps=float(text.get("rms_norm_eps", 1e-6)),
        num_attention_heads=int(text["num_attention_heads"]),
        num_key_value_heads=int(text["num_key_value_heads"]),
        head_dim=head_dim,
        rope_theta=float(rope.get("rope_theta", 10000.0)),
        rotary_dim=int(head_dim * partial_rotary_factor),
        is_linear_per_layer=is_linear_per_layer,
        linear=Qwen3_5LinearAttention(
            num_key_heads=int(text["linear_num_key_heads"]),
            num_value_heads=int(text["linear_num_value_heads"]),
            key_head_dim=int(text["linear_key_head_dim"]),
            value_head_dim=int(text["linear_value_head_dim"]),
            conv_kernel_dim=int(text["linear_conv_kernel_dim"]),
        ),
        num_experts=int(text["num_experts"]),
        experts_per_token=int(text["num_experts_per_tok"]),
        moe_intermediate_size=int(text["moe_intermediate_size"]),
        shared_expert_intermediate_size=int(text["shared_expert_intermediate_size"]),
        # The reference implementation's own default, and it changes the
        # routed scores.
        norm_topk_prob=bool(text.get("norm_topk_prob", True)),
        expert_quant=default_quant,
        router_quant=_quant_override(quant, ".mlp.gate", default_quant),
        shared_gate_quant=_quant_override(
            quant, ".mlp.shared_expert_gate", default_quant
        ),
        other_quant=default_quant,
        quant_overrides=_quant_overrides(quant, default_quant),
    )
