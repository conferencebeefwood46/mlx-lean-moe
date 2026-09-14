"""Real-weight validation against `mlx_lm`'s own modules loaded with the
same weights off the same checkpoint.

Where the synthetic tests prove the algorithms, these prove the checkpoint
is read correctly, on one layer of each kind. Skips when it is not there.
"""

import json

import mlx.core as mx
import pytest
from conftest import QWEN3_5_MODEL_DIR

from mlx_lean_moe.cache.kv_cache import GrowingKVCache
from mlx_lean_moe.cache.recurrent_cache import RecurrentCache
from mlx_lean_moe.config import model_config_from_hf
from mlx_lean_moe.model.qwen3_5.attention import Qwen3_5Attention
from mlx_lean_moe.model.qwen3_5.linear_attention import Qwen3_5LinearAttention
from mlx_lean_moe.weights.expert_loader import TensorStreamer
from mlx_lean_moe.weights.safetensors_index import build_index

qwen3_next_ref = pytest.importorskip("mlx_lm.models.qwen3_next")
qwen3_5_ref = pytest.importorskip("mlx_lm.models.qwen3_5")

pytestmark = pytest.mark.skipif(
    not (QWEN3_5_MODEL_DIR / "model.safetensors.index.json").exists()
    or not list(QWEN3_5_MODEL_DIR.glob("model-*.safetensors")),
    reason="the qwen3_5 validation checkpoint is not fully downloaded",
)

PREFIX = "language_model.model"
LINEAR_LAYER = 0
FULL_LAYER = 3


@pytest.fixture(scope="module")
def config():
    return model_config_from_hf(
        json.loads((QWEN3_5_MODEL_DIR / "config.json").read_text())
    )


@pytest.fixture(scope="module")
def streamer():
    s = TensorStreamer(QWEN3_5_MODEL_DIR, build_index(QWEN3_5_MODEL_DIR))
    yield s
    s.close()


def read_quantized(streamer: TensorStreamer, name: str) -> dict[str, mx.array]:
    """Read one quantized tensor's ``weight``/``scales``/``biases`` triple
    off ``name``'s prefix (e.g. ``"model.layers.0.mlp.gate"``)."""
    return {
        "weight": streamer.read_tensor(f"{name}.weight"),
        "scales": streamer.read_tensor(f"{name}.scales"),
        "biases": streamer.read_tensor(f"{name}.biases"),
    }


def _dequantize(tensors, quant) -> mx.array:
    """The reference's dense weight at full precision: `mx.dequantize`
    returns whatever dtype its scales carry, bfloat16 included."""
    return mx.dequantize(
        tensors["weight"],
        scales=tensors["scales"].astype(mx.float32),
        biases=tensors["biases"].astype(mx.float32),
        group_size=quant.group_size,
        bits=quant.bits,
    )


def _ref_args(config):
    return qwen3_5_ref.TextModelArgs(
        model_type="qwen3_5",
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        rms_norm_eps=config.rms_norm_eps,
        linear_num_key_heads=config.linear.num_key_heads,
        linear_num_value_heads=config.linear.num_value_heads,
        linear_key_head_dim=config.linear.key_head_dim,
        linear_value_head_dim=config.linear.value_head_dim,
        linear_conv_kernel_dim=config.linear.conv_kernel_dim,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": config.rope_theta,
            "partial_rotary_factor": config.rotary_dim / config.head_dim,
        },
    )


def test_config_matches_the_real_checkpoints_shape(config):
    assert config.num_layers == 40
    assert sum(config.is_linear_per_layer) == 30  # three quarters are recurrent
    assert config.is_linear_per_layer[LINEAR_LAYER] is True
    assert config.is_linear_per_layer[FULL_LAYER] is False
    assert config.num_experts == 256
    assert config.experts_per_token == 8
    assert config.hidden_size == 2048
    assert config.rotary_dim == 64 and config.head_dim == 256


@pytest.mark.parametrize("seq_len", [4, 257])
def test_linear_attention_matches_mlx_lm_on_real_weights(config, streamer, seq_len):
    prefix = f"{PREFIX}.layers.{LINEAR_LAYER}.linear_attn"
    quant = config.other_quant
    tensors = {
        name: read_quantized(streamer, f"{prefix}.{name}")
        for name in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj")
    }
    conv_weight = streamer.read_tensor(f"{prefix}.conv1d.weight")
    A_log = streamer.read_tensor(f"{prefix}.A_log")
    dt_bias = streamer.read_tensor(f"{prefix}.dt_bias")
    norm_weight = streamer.read_tensor(f"{prefix}.norm.weight")

    ours = Qwen3_5LinearAttention(
        config,
        in_proj_qkv=tensors["in_proj_qkv"],
        in_proj_z=tensors["in_proj_z"],
        in_proj_a=tensors["in_proj_a"],
        in_proj_b=tensors["in_proj_b"],
        conv1d_weight=conv_weight,
        A_log=A_log,
        dt_bias=dt_bias,
        norm_weight=norm_weight,
        out_proj=tensors["out_proj"],
    )

    ref = qwen3_5_ref.GatedDeltaNet(_ref_args(config))
    for name in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"):
        getattr(ref, name).weight = _dequantize(tensors[name], quant)
    ref.conv1d.weight = conv_weight
    ref.A_log = A_log
    ref.dt_bias = dt_bias
    ref.norm.weight = norm_weight

    mx.random.seed(0)
    x = mx.random.normal((seq_len, config.hidden_size))
    cache = RecurrentCache(
        conv_kernel_dim=config.linear.conv_kernel_dim,
        conv_dim=config.linear.conv_dim,
        num_value_heads=config.linear.num_value_heads,
        value_head_dim=config.linear.value_head_dim,
        key_head_dim=config.linear.key_head_dim,
    )

    ref_cache = pytest.importorskip("mlx_lm.models.cache").ArraysCache(size=2)
    got = ours(x, cache)
    expected = ref(x[None], mask=None, cache=ref_cache)[0]
    mx.eval(got, expected)
    assert mx.allclose(got, expected, rtol=1e-3, atol=1e-3).item(), float(
        mx.max(mx.abs(got - expected)).item()
    )

    # Continue from actual checkpoint activations, testing both convolution
    # history and recurrent state after a multi-chunk-sized prompt.
    for length in (1, 3):
        x = mx.random.normal((length, config.hidden_size))
        got, expected = ours(x, cache), ref(x[None], mask=None, cache=ref_cache)[0]
        mx.eval(got, expected)
        assert mx.allclose(got, expected, rtol=1e-3, atol=1e-3).item()
        for actual, reference in zip(cache.state(), ref_cache.state):
            assert mx.allclose(actual, reference[0], rtol=1e-3, atol=1e-3).item()


def test_full_attention_matches_mlx_lm_on_real_weights(config, streamer):
    prefix = f"{PREFIX}.layers.{FULL_LAYER}.self_attn"
    quant = config.other_quant
    tensors = {
        name: read_quantized(streamer, f"{prefix}.{name}")
        for name in ("q_proj", "k_proj", "v_proj", "o_proj")
    }
    q_norm = streamer.read_tensor(f"{prefix}.q_norm.weight")
    k_norm = streamer.read_tensor(f"{prefix}.k_norm.weight")

    ours = Qwen3_5Attention(
        config,
        q_proj=tensors["q_proj"],
        q_norm=q_norm,
        k_proj=tensors["k_proj"],
        k_norm=k_norm,
        v_proj=tensors["v_proj"],
        o_proj=tensors["o_proj"],
    )

    ref = qwen3_next_ref.Qwen3NextAttention(_ref_args(config))
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        getattr(ref, name).weight = _dequantize(tensors[name], quant)
    ref.q_norm.weight = q_norm
    ref.k_norm.weight = k_norm

    mx.random.seed(1)
    x = mx.random.normal((4, config.hidden_size))
    cache = GrowingKVCache(
        max_context=16,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=mx.float32,
    )

    got = ours(x, cache)
    mlx_lm_cache = pytest.importorskip("mlx_lm.models.cache")
    expected = ref(x[None], mask="causal", cache=mlx_lm_cache.KVCache())[0]
    mx.eval(got, expected)
    assert mx.allclose(got, expected, rtol=1e-3, atol=1e-3).item(), float(
        mx.max(mx.abs(got - expected)).item()
    )


def test_q_proj_really_is_double_width_for_the_output_gate(config, streamer):
    """The gate has no tensor of its own: it is the second half of q_proj's
    output, confirmed here on the real checkpoint."""
    q_proj = read_quantized(streamer, f"{PREFIX}.layers.{FULL_LAYER}.self_attn.q_proj")
    k_proj = read_quantized(streamer, f"{PREFIX}.layers.{FULL_LAYER}.self_attn.k_proj")
    assert q_proj["scales"].shape[0] == config.num_attention_heads * config.head_dim * 2
    assert k_proj["scales"].shape[0] == config.num_key_value_heads * config.head_dim
    prefix = f"{PREFIX}.layers.{FULL_LAYER}.self_attn"
    assert f"{prefix}.gate_proj.weight" not in streamer.index
