"""The gated-delta recurrence behind qwen3_5's linear-attention layers: one
step per value head over a ``(value_head_dim, key_head_dim)`` state.

Ported from `mlx_lm.models.gated_delta`'s ops reference and cross-checked
against it in tests. Single sequence, no batch axis.
"""

import mlx.core as mx
import mlx.nn as nn

from mlx_lean_moe.model.qwen3_5._gated_delta_metal import gated_delta_scan_metal


def forget_gate(a: mx.array, A_log: mx.array, dt_bias: mx.array) -> mx.array:
    """Per-value-head decay in (0, 1], for one position. In float32: it
    feeds a product accumulated over the whole sequence."""

    return mx.exp(
        -mx.exp(A_log.astype(mx.float32))
        * nn.softplus(a.astype(mx.float32) + dt_bias.astype(mx.float32))
    )


def gated_delta_step(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
) -> tuple[mx.array, mx.array]:
    """One recurrent step. ``q``/``k`` are ``(num_value_heads, key_head_dim)``,
    already repeated out from their own smaller head count."""

    state = state * g[:, None, None]
    read = (state * k[:, None, :]).sum(axis=-1)
    delta = (v - read) * beta[:, None]
    state = state + k[:, None, :] * delta[:, :, None]
    y = (state * q[:, None, :]).sum(axis=-1)

    return y, state


def gated_delta_scan(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    *,
    use_kernel: bool = True,
) -> tuple[mx.array, mx.array]:
    """Runs :func:`gated_delta_step` over ``L`` positions in order, on Metal
    where supported; ``use_kernel=False`` forces the ops path."""

    if q.ndim != 3 or k.shape != q.shape or v.ndim != 3:
        raise ValueError(
            "q/k must have matching (L, key_heads, key_dim) shapes; v must be 3-D"
        )

    length, key_heads, key_dim = q.shape
    _, value_heads, value_dim = v.shape

    if min(length, key_heads, key_dim, value_heads, value_dim) <= 0:
        raise ValueError(
            "gated_delta_scan requires nonempty sequence and head dimensions"
        )

    if v.shape[0] != length or value_heads % key_heads:
        raise ValueError(
            "v must match q's sequence length and have a multiple of its head count"
        )

    if g.shape != (length, value_heads) or beta.shape != g.shape:
        raise ValueError("g and beta must have shape (L, value_heads)")

    if state.shape != (value_heads, value_dim, key_dim):
        raise ValueError("state must have shape (value_heads, value_dim, key_dim)")

    if (
        use_kernel
        and mx.default_device() == mx.gpu
        and mx.metal.is_available()
        and key_dim in (32, 64, 128, 256)
        and all(a.dtype == mx.float32 for a in (q, k, v, g, beta, state))
    ):
        return gated_delta_scan_metal(q, k, v, g, beta, state)

    return _gated_delta_scan_ops(q, k, v, g, beta, state)


def _gated_delta_scan_ops(q, k, v, g, beta, state) -> tuple[mx.array, mx.array]:
    """Reference recurrence, also used for unsupported Metal specializations."""

    num_value_heads = v.shape[-2]
    repeat = num_value_heads // q.shape[-2]

    if repeat > 1:
        q = mx.repeat(q, repeat, axis=-2)
        k = mx.repeat(k, repeat, axis=-2)

    outputs = []

    for t in range(q.shape[0]):
        y, state = gated_delta_step(q[t], k[t], v[t], g[t], beta[t], state)
        outputs.append(y)

    return mx.stack(outputs), state
