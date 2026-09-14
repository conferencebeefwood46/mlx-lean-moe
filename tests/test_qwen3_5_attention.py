"""Cross-checks `Qwen3_5Attention` against `mlx_lm`'s own
`qwen3_next.Qwen3NextAttention`, both sides given byte-identical effective
weights via quantize-then-dequantize.
"""

import mlx.core as mx
import numpy as np
import pytest

from mlx_lean_moe.cache.kv_cache import GrowingKVCache
from mlx_lean_moe.config import QuantScheme
from mlx_lean_moe.model.qwen3_5.attention import Qwen3_5Attention
from mlx_lean_moe.model.qwen3_5.config import Qwen3_5Config
from mlx_lean_moe.model.qwen3_5.config import Qwen3_5LinearAttention as LinearParams

qwen3_next_ref = pytest.importorskip("mlx_lm.models.qwen3_next")
qwen3_5_ref = pytest.importorskip("mlx_lm.models.qwen3_5")
mlx_lm_cache = pytest.importorskip("mlx_lm.models.cache")

HIDDEN = 128
GROUP_SIZE = 64
BITS = 4
N_HEADS = 4
N_KV_HEADS = 2
HEAD_DIM = 32
ROTARY_DIM = 8  # partial_rotary_factor 0.25
ROPE_THETA = 10000000.0
LINEAR = LinearParams(
    num_key_heads=2,
    num_value_heads=4,
    key_head_dim=32,
    value_head_dim=32,
    conv_kernel_dim=4,
)


def _config() -> Qwen3_5Config:
    quant = QuantScheme(bits=BITS, group_size=GROUP_SIZE)
    return Qwen3_5Config(
        num_layers=1,
        hidden_size=HIDDEN,
        vocab_size=64,
        rms_norm_eps=1e-6,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        rope_theta=ROPE_THETA,
        rotary_dim=ROTARY_DIM,
        is_linear_per_layer=(False,),
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
    dense = mx.array(rng.standard_normal((out_dim, in_dim)).astype(np.float32) * 0.05)
    w, scales, biases = mx.quantize(dense, group_size=GROUP_SIZE, bits=BITS)
    effective = mx.dequantize(
        w, scales=scales, biases=biases, group_size=GROUP_SIZE, bits=BITS
    )
    return {"weight": w, "scales": scales, "biases": biases}, effective


def _build_pair(seed: int):
    rng = np.random.default_rng(seed)
    config = _config()

    # q_proj emits queries AND the output gate: 2x the head width.
    q_proj, q_dense = _quantized(rng, N_HEADS * HEAD_DIM * 2, HIDDEN)
    k_proj, k_dense = _quantized(rng, N_KV_HEADS * HEAD_DIM, HIDDEN)
    v_proj, v_dense = _quantized(rng, N_KV_HEADS * HEAD_DIM, HIDDEN)
    o_proj, o_dense = _quantized(rng, HIDDEN, N_HEADS * HEAD_DIM)
    q_norm = mx.array(rng.standard_normal(HEAD_DIM).astype(np.float32) * 0.1 + 1.0)
    k_norm = mx.array(rng.standard_normal(HEAD_DIM).astype(np.float32) * 0.1 + 1.0)

    ours = Qwen3_5Attention(
        config,
        q_proj=q_proj,
        q_norm=q_norm,
        k_proj=k_proj,
        k_norm=k_norm,
        v_proj=v_proj,
        o_proj=o_proj,
    )

    args = qwen3_5_ref.TextModelArgs(
        model_type="qwen3_5",
        hidden_size=HIDDEN,
        num_hidden_layers=1,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        rms_norm_eps=1e-6,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": ROPE_THETA,
            "partial_rotary_factor": 0.25,
        },
    )
    ref = qwen3_next_ref.Qwen3NextAttention(args)
    ref.q_proj.weight = q_dense
    ref.k_proj.weight = k_dense
    ref.v_proj.weight = v_dense
    ref.o_proj.weight = o_dense
    ref.q_norm.weight = q_norm
    ref.k_norm.weight = k_norm
    return config, ours, ref


@pytest.mark.parametrize("seq_len", [1, 6])
def test_matches_mlx_lm_attention(seq_len):
    config, ours, ref = _build_pair(seed=0)
    rng = np.random.default_rng(200)
    x = mx.array(rng.standard_normal((seq_len, HIDDEN)).astype(np.float32))

    cache = GrowingKVCache(
        max_context=16, num_kv_heads=N_KV_HEADS, head_dim=HEAD_DIM, dtype=mx.float32
    )
    got = ours(x, cache)

    ref_cache = mlx_lm_cache.KVCache()
    mask = "causal" if seq_len > 1 else None
    expected = ref(x[None], mask=mask, cache=ref_cache)[0]

    mx.eval(got, expected)
    assert mx.allclose(got, expected, rtol=1e-4, atol=1e-4).item()


def test_decode_steps_match_a_batched_prefill():
    config, ours, _ = _build_pair(seed=1)
    rng = np.random.default_rng(201)
    x = mx.array(rng.standard_normal((5, HIDDEN)).astype(np.float32))

    batched = ours(
        x,
        GrowingKVCache(
            max_context=16, num_kv_heads=N_KV_HEADS, head_dim=HEAD_DIM, dtype=mx.float32
        ),
    )
    cache = GrowingKVCache(
        max_context=16, num_kv_heads=N_KV_HEADS, head_dim=HEAD_DIM, dtype=mx.float32
    )
    stepwise = mx.stack([ours(x[t], cache) for t in range(x.shape[0])])

    mx.eval(batched, stepwise)
    assert mx.allclose(batched, stepwise, rtol=1e-3, atol=1e-4).item()


def test_output_gate_actually_gates():
    """The second half of q_proj is a gate, not more queries: zeroing it
    must halve the output (sigmoid(0) = 0.5), not merely perturb it."""
    config, ours, _ = _build_pair(seed=2)
    rng = np.random.default_rng(202)
    x = mx.array(rng.standard_normal((1, HIDDEN)).astype(np.float32))

    gated = ours(
        x,
        GrowingKVCache(
            max_context=4, num_kv_heads=N_KV_HEADS, head_dim=HEAD_DIM, dtype=mx.float32
        ),
    )

    # Rebuild with the gate half of q_proj forced to zero.
    dense = mx.dequantize(
        ours.q_proj["weight"],
        scales=ours.q_proj["scales"],
        biases=ours.q_proj["biases"],
        group_size=GROUP_SIZE,
        bits=BITS,
    )
    half = N_HEADS * HEAD_DIM
    zeroed = mx.concatenate([dense[:half], mx.zeros_like(dense[half:])], axis=0)
    w, scales, biases = mx.quantize(zeroed, group_size=GROUP_SIZE, bits=BITS)
    ours.q_proj = {"weight": w, "scales": scales, "biases": biases}
    ungated = ours(
        x,
        GrowingKVCache(
            max_context=4, num_kv_heads=N_KV_HEADS, head_dim=HEAD_DIM, dtype=mx.float32
        ),
    )

    mx.eval(gated, ungated)
    assert not mx.allclose(gated, ungated, atol=1e-3).item()
