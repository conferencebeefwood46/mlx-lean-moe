"""Partial-rotary RoPE, pairing ``i`` with ``i + rotary_dim // 2``.

The other convention pairs ``i`` with ``i + head_dim // 2``, as
`mlx_lm.models.rope_utils.ProportionalRoPE` does.
"""

import mlx.core as mx


def apply_leading_rope(
    x: mx.array, rotary_dim: int, base: float, offset: int | mx.array
) -> mx.array:
    """Rotates the leading ``rotary_dim`` dimensions of ``x``, shaped
    ``(B, n_heads, seq_len, head_dim)``, and passes the rest through."""
    return mx.fast.rope(
        x, rotary_dim, traditional=False, base=base, scale=1.0, offset=offset
    )
