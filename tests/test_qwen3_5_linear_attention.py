"""Cross-checks the whole gated-delta layer against `mlx_lm`'s own
`qwen3_5.GatedDeltaNet`.

Weights are quantized and then dequantized into the reference, so its plain
matmul and this project's `mx.quantized_matmul` see the same numbers.
"""

import mlx.core as mx
import numpy as np
import pytest

from mlx_lean_moe.cache.recurrent_cache import RecurrentCache
from mlx_lean_moe.config import QuantScheme
from mlx_lean_moe.model.qwen3_5.config import Qwen3_5Config
from mlx_lean_moe.model.qwen3_5.config import Qwen3_5LinearAttention as LinearParams
from mlx_lean_moe.model.qwen3_5.linear_attention import Qwen3_5LinearAttention

qwen3_5_ref = pytest.importorskip("mlx_lm.models.qwen3_5")

HIDDEN = 128
GROUP_SIZE = 64
BITS = 4
LINEAR = LinearParams(num_key_heads=2, num_value_heads=4, key_head_dim=32, value_head_dim=32, conv_kernel_dim=4)


def _config() -> Qwen3_5Config:
    quant = QuantScheme(bits=BITS, group_size=GROUP_SIZE)
    return Qwen3_5Config(
        num_layers=1,
        hidden_size=HIDDEN,
        vocab_size=64,
        rms_norm_eps=1e-6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        rope_theta=10000.0,
        rotary_dim=8,
        is_linear_per_layer=(True,),
        linear=LINEAR,
        num_experts=4,
        experts_per_token=2,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=64,
        norm_topk_prob=True,
        expert_quant=quant,
        router_quant=quant,
        shared_gate_quant=quant,
        other_quant=quant,
    )


def _quantized(rng, out_dim: int, in_dim: int):
    """A random weight, quantized: both this project's {weight,scales,biases}
    and the dequantized dense matrix the reference gets."""
    dense = mx.array(rng.standard_normal((out_dim, in_dim)).astype(np.float32) * 0.05)
    w, scales, biases = mx.quantize(dense, group_size=GROUP_SIZE, bits=BITS)
    effective = mx.dequantize(w, scales=scales, biases=biases, group_size=GROUP_SIZE, bits=BITS)
    return {"weight": w, "scales": scales, "biases": biases}, effective


def _build_pair(seed: int):
    rng = np.random.default_rng(seed)
    config = _config()

    qkv_dim = LINEAR.conv_dim
    in_qkv, in_qkv_dense = _quantized(rng, qkv_dim, HIDDEN)
    in_z, in_z_dense = _quantized(rng, LINEAR.value_dim, HIDDEN)
    in_a, in_a_dense = _quantized(rng, LINEAR.num_value_heads, HIDDEN)
    in_b, in_b_dense = _quantized(rng, LINEAR.num_value_heads, HIDDEN)
    out_proj, out_proj_dense = _quantized(rng, HIDDEN, LINEAR.value_dim)

    conv_weight = mx.array(rng.standard_normal((qkv_dim, LINEAR.conv_kernel_dim, 1)).astype(np.float32) * 0.2)
    A_log = mx.array(rng.standard_normal(LINEAR.num_value_heads).astype(np.float32))
    dt_bias = mx.array(rng.standard_normal(LINEAR.num_value_heads).astype(np.float32))
    norm_weight = mx.array(rng.standard_normal(LINEAR.value_head_dim).astype(np.float32) * 0.1 + 1.0)

    ours = Qwen3_5LinearAttention(
        config,
        in_proj_qkv=in_qkv,
        in_proj_z=in_z,
        in_proj_a=in_a,
        in_proj_b=in_b,
        conv1d_weight=conv_weight,
        A_log=A_log,
        dt_bias=dt_bias,
        norm_weight=norm_weight,
        out_proj=out_proj,
    )

    args = qwen3_5_ref.TextModelArgs(
        model_type="qwen3_5",
        hidden_size=HIDDEN,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        rms_norm_eps=1e-6,
        linear_num_key_heads=LINEAR.num_key_heads,
        linear_num_value_heads=LINEAR.num_value_heads,
        linear_key_head_dim=LINEAR.key_head_dim,
        linear_value_head_dim=LINEAR.value_head_dim,
        linear_conv_kernel_dim=LINEAR.conv_kernel_dim,
    )
    ref = qwen3_5_ref.GatedDeltaNet(args)
    ref.in_proj_qkv.weight = in_qkv_dense
    ref.in_proj_z.weight = in_z_dense
    ref.in_proj_a.weight = in_a_dense
    ref.in_proj_b.weight = in_b_dense
    ref.out_proj.weight = out_proj_dense
    ref.conv1d.weight = conv_weight
    ref.A_log = A_log
    ref.dt_bias = dt_bias
    ref.norm.weight = norm_weight
    return config, ours, ref


def _fresh_cache(config: Qwen3_5Config) -> RecurrentCache:
    return RecurrentCache(
        conv_kernel_dim=LINEAR.conv_kernel_dim,
        conv_dim=LINEAR.conv_dim,
        num_value_heads=LINEAR.num_value_heads,
        value_head_dim=LINEAR.value_head_dim,
        key_head_dim=LINEAR.key_head_dim,
    )


@pytest.mark.parametrize("seq_len", [1, 6])
def test_matches_mlx_lm_gated_delta_net(seq_len):
    config, ours, ref = _build_pair(seed=0)
    rng = np.random.default_rng(100)
    x = mx.array(rng.standard_normal((seq_len, HIDDEN)).astype(np.float32))

    got = ours(x, _fresh_cache(config))
    expected = ref(x[None], mask=None, cache=None)[0]

    mx.eval(got, expected)
    assert mx.allclose(got, expected, rtol=1e-4, atol=1e-4).item()


def test_decoding_step_by_step_matches_one_batched_pass():
    """Both halves of the state, the convolution's tail and the SSM matrix,
    have to carry across calls exactly."""
    config, ours, _ = _build_pair(seed=1)
    rng = np.random.default_rng(101)
    x = mx.array(rng.standard_normal((5, HIDDEN)).astype(np.float32))

    batched = ours(x, _fresh_cache(config))

    cache = _fresh_cache(config)
    stepwise = mx.stack([ours(x[t], cache) for t in range(x.shape[0])])

    mx.eval(batched, stepwise)
    assert mx.allclose(batched, stepwise, rtol=1e-4, atol=1e-5).item()


def test_cache_state_is_constant_in_sequence_length():
    """The whole point of these layers: feeding more tokens must not grow
    the state one byte."""
    config, ours, _ = _build_pair(seed=2)
    rng = np.random.default_rng(102)
    cache = _fresh_cache(config)
    shapes = []
    for chunk in range(4):
        x = mx.array(rng.standard_normal((3, HIDDEN)).astype(np.float32))
        ours(x, cache)
        conv_state, ssm_state = cache.state()
        mx.eval(conv_state, ssm_state)
        shapes.append((conv_state.shape, ssm_state.shape))

    assert len(set(shapes)) == 1
    assert cache.size == 12
