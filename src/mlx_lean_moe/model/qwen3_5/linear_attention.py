"""qwen3_5's gated-delta linear attention layer (``linear_attn``).

Most of the model's layers are these: rather than a key/value history, each
keeps a fixed-size state updated by
:mod:`mlx_lean_moe.model.qwen3_5.gated_delta`'s delta rule. Reproduces
`mlx_lm`'s ``qwen3_5.GatedDeltaNet``.
"""

import mlx.core as mx
import mlx.nn as nn

from mlx_lean_moe.cache.recurrent_cache import RecurrentCache
from mlx_lean_moe.model.moe_block import LinearWeights, QuantizedTensor, quantized_linear
from mlx_lean_moe.model.qwen3_5.config import Qwen3_5Config
from mlx_lean_moe.model.qwen3_5.gated_delta import forget_gate, gated_delta_scan


def _gated_rms_norm(x: mx.array, gate: mx.array, weight: mx.array, eps: float) -> mx.array:
    """``silu(gate) * rms_norm(x, weight)``, the product taken in float32 as
    the reference's own precise path does."""
    normed = mx.fast.rms_norm(x, weight, eps)
    return (nn.silu(gate.astype(mx.float32)) * normed.astype(mx.float32)).astype(x.dtype)


class Qwen3_5LinearAttention:
    def __init__(
        self,
        config: Qwen3_5Config,
        in_proj_qkv: LinearWeights | QuantizedTensor,
        in_proj_z: LinearWeights | QuantizedTensor,
        in_proj_a: LinearWeights | QuantizedTensor,
        in_proj_b: LinearWeights | QuantizedTensor,
        conv1d_weight: mx.array,
        A_log: mx.array,
        dt_bias: mx.array,
        norm_weight: mx.array,
        out_proj: LinearWeights | QuantizedTensor,
    ) -> None:
        self.config = config
        self.linear = config.linear
        self.in_proj_qkv = in_proj_qkv
        self.in_proj_z = in_proj_z
        self.in_proj_a = in_proj_a
        self.in_proj_b = in_proj_b
        self.conv1d_weight = conv1d_weight
        self.A_log = A_log
        self.dt_bias = dt_bias
        self.norm_weight = norm_weight
        self.out_proj = out_proj

    def _convolve(self, qkv: mx.array, cache: RecurrentCache) -> tuple[mx.array, mx.array]:
        """Depthwise causal convolution over ``(L, conv_dim)``, prefixed with
        the previous call's tail. Returns the output and the new state."""
        kernel = self.linear.conv_kernel_dim
        conv_state, _ = cache.state()
        conv_input = mx.concatenate([conv_state.astype(qkv.dtype), qkv], axis=0)
        out = mx.conv1d(conv_input[None], self.conv1d_weight.astype(conv_input.dtype), groups=self.linear.conv_dim)[0]
        # From the front, not a negative index: a kernel of 1 must keep
        # nothing rather than everything.
        tail_start = conv_input.shape[0] - (kernel - 1)
        return nn.silu(out), conv_input[tail_start:]

    def __call__(self, x: mx.array, cache: RecurrentCache) -> mx.array:
        """``x``: ``(hidden_size,)`` for one decode step, or
        ``(L, hidden_size)`` for a whole prompt. Returns the same rank."""
        if x.ndim not in (1, 2):
            raise ValueError(f"Qwen3_5LinearAttention expects 1-D or 2-D x, got shape {x.shape}")
        squeeze = x.ndim == 1
        rows = x[None, :] if squeeze else x
        seq_len = rows.shape[0]
        linear = self.linear
        quant = self.config.other_quant

        qkv = quantized_linear(rows, self.in_proj_qkv, quant)
        z = quantized_linear(rows, self.in_proj_z, quant).reshape(
            seq_len, linear.num_value_heads, linear.value_head_dim
        )
        a = quantized_linear(rows, self.in_proj_a, quant)
        b = quantized_linear(rows, self.in_proj_b, quant)

        conv_out, new_conv_state = self._convolve(qkv, cache)
        q, k, v = mx.split(conv_out, [linear.key_dim, 2 * linear.key_dim], axis=-1)
        q = q.reshape(seq_len, linear.num_key_heads, linear.key_head_dim)
        k = k.reshape(seq_len, linear.num_key_heads, linear.key_head_dim)
        v = v.reshape(seq_len, linear.num_value_heads, linear.value_head_dim)

        # 1e-6 rather than the model's rms_norm_eps: the reference hardcodes
        # it here, and the q/k rescaling below is deliberately asymmetric.
        inv_scale = linear.key_head_dim**-0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

        g = forget_gate(a, self.A_log, self.dt_bias)
        beta = mx.sigmoid(b)

        _, ssm_state = cache.state()
        y, ssm_state = gated_delta_scan(q, k, v, g, beta, ssm_state)
        cache.update(new_conv_state, ssm_state, advance=seq_len)

        out = _gated_rms_norm(y.astype(z.dtype), z, self.norm_weight, self.config.rms_norm_eps)
        out = quantized_linear(out.reshape(seq_len, -1), self.out_proj, quant)
        return out[0] if squeeze else out
