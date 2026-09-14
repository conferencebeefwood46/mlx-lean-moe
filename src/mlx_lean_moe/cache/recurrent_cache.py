"""Fixed-size state for a recurrent (linear-attention) layer.

Two pieces: the last ``kernel_size - 1`` convolution inputs, and a
per-value-head ``(value_head_dim, key_head_dim)`` matrix. Both are constant
in context length, unlike :mod:`mlx_lean_moe.cache.kv_cache`.
"""

import mlx.core as mx


class RecurrentCache:
    """One gated-delta layer's rolling state, mutated in place by the layer
    for the life of a generation."""

    def __init__(
        self,
        conv_kernel_dim: int,
        conv_dim: int,
        num_value_heads: int,
        value_head_dim: int,
        key_head_dim: int,
        dtype: mx.Dtype = mx.float32,
    ) -> None:
        if conv_kernel_dim < 1:
            raise ValueError(f"conv_kernel_dim must be positive, got {conv_kernel_dim}")
        self.conv_state = mx.zeros((conv_kernel_dim - 1, conv_dim), dtype=dtype)
        # float32 whatever the checkpoint's dtype, as the reference does via
        # `mamba_ssm_dtype`: this accumulates over the whole sequence.
        self.ssm_state = mx.zeros(
            (num_value_heads, value_head_dim, key_head_dim), dtype=mx.float32
        )
        self.size = 0

    def state(self) -> tuple[mx.array, mx.array]:
        return self.conv_state, self.ssm_state

    def update(self, conv_state: mx.array, ssm_state: mx.array, advance: int) -> None:
        self.conv_state = conv_state
        self.ssm_state = ssm_state
        self.size += advance
