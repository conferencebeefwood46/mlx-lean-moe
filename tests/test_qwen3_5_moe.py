"""Cross-checks qwen3_5's MoE block against `mlx_lm`'s own
`qwen3_next.Qwen3NextSparseMoeBlock`.

The routed experts stream, so the fixture writes a tiny stacked checkpoint
to disk and loads through the real path.
"""

import mlx.core as mx
import numpy as np
import pytest
from safetensors.numpy import save_file

from mlx_lean_moe.config import QuantScheme
from mlx_lean_moe.model.qwen3_5.config import Qwen3_5Config
from mlx_lean_moe.model.qwen3_5.config import Qwen3_5LinearAttention as LinearParams
from mlx_lean_moe.model.qwen3_5.mlp import (
    Qwen3_5Experts,
    Qwen3_5Router,
    Qwen3_5SharedExpert,
    swiglu_mlp,
)
from mlx_lean_moe.weights.expert_loader import CachedExpertLoader
from mlx_lean_moe.weights.safetensors_index import build_index
from mlx_lean_moe.weights.stacked_expert_loader import StackedExpertStreamer

qwen3_next_ref = pytest.importorskip("mlx_lm.models.qwen3_next")
qwen3_5_ref = pytest.importorskip("mlx_lm.models.qwen3_5")

HIDDEN = 128
GROUP_SIZE = 64
BITS = 4
NUM_EXPERTS = 8
TOP_K = 3
MOE_INTER = 64
SHARED_INTER = 128
PREFIX = "language_model.model"
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
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        rope_theta=10000.0,
        rotary_dim=8,
        is_linear_per_layer=(False,),
        linear=LINEAR,
        num_experts=NUM_EXPERTS,
        experts_per_token=TOP_K,
        moe_intermediate_size=MOE_INTER,
        shared_expert_intermediate_size=SHARED_INTER,
        norm_topk_prob=True,
        expert_quant=quant,
        router_quant=quant,
        shared_gate_quant=quant,
        other_quant=quant,
    )


def _quantize(dense: mx.array):
    w, scales, biases = mx.quantize(dense, group_size=GROUP_SIZE, bits=BITS)
    effective = mx.dequantize(
        w, scales=scales, biases=biases, group_size=GROUP_SIZE, bits=BITS
    )
    return {"weight": w, "scales": scales, "biases": biases}, effective


def _random(rng, *shape) -> mx.array:
    return mx.array(rng.standard_normal(shape).astype(np.float32) * 0.05)


@pytest.fixture
def moe_pair(tmp_path):
    """This project's MoE pieces and the reference block from the same
    weights, the routed experts written to a stacked checkpoint."""
    rng = np.random.default_rng(0)
    config = _config()

    router_t, router_dense = _quantize(_random(rng, NUM_EXPERTS, HIDDEN))
    shared = {}
    shared_dense = {}
    for name, out_dim, in_dim in (
        ("gate_proj", SHARED_INTER, HIDDEN),
        ("up_proj", SHARED_INTER, HIDDEN),
        ("down_proj", HIDDEN, SHARED_INTER),
    ):
        shared[name], shared_dense[name] = _quantize(_random(rng, out_dim, in_dim))
    shared_gate_t, shared_gate_dense = _quantize(_random(rng, 1, HIDDEN))

    # One stacked (num_experts, out, in) tensor per projection, quantized
    # per expert then stacked, as a real checkpoint stores them.
    stacked_files: dict[str, np.ndarray] = {}
    expert_dense: dict[str, list[mx.array]] = {}
    for name, out_dim, in_dim in (
        ("gate_proj", MOE_INTER, HIDDEN),
        ("up_proj", MOE_INTER, HIDDEN),
        ("down_proj", HIDDEN, MOE_INTER),
    ):
        packed, scales, biases, dense = [], [], [], []
        for _ in range(NUM_EXPERTS):
            tensors, effective = _quantize(_random(rng, out_dim, in_dim))
            packed.append(np.array(tensors["weight"]))
            scales.append(np.array(tensors["scales"]))
            biases.append(np.array(tensors["biases"]))
            dense.append(effective)
        base = f"{PREFIX}.layers.0.mlp.switch_mlp.{name}"
        stacked_files[f"{base}.weight"] = np.stack(packed)
        stacked_files[f"{base}.scales"] = np.stack(scales)
        stacked_files[f"{base}.biases"] = np.stack(biases)
        expert_dense[name] = dense

    save_file(stacked_files, str(tmp_path / "model.safetensors"))
    index = build_index(tmp_path, use_cache=False)
    streamer = StackedExpertStreamer(
        tmp_path,
        index,
        num_experts=NUM_EXPERTS,
        layer_prefix=f"{PREFIX}.layers",
        stack_name="mlp.switch_mlp",
    )
    loader = CachedExpertLoader(streamer, max_size=NUM_EXPERTS)

    ours = {
        "router": Qwen3_5Router(config, router_t),
        "experts": Qwen3_5Experts(config, 0, loader),
        "shared": Qwen3_5SharedExpert(config, shared, shared_gate_t),
    }

    args = qwen3_5_ref.TextModelArgs(
        model_type="qwen3_5",
        hidden_size=HIDDEN,
        num_hidden_layers=1,
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=TOP_K,
        moe_intermediate_size=MOE_INTER,
        shared_expert_intermediate_size=SHARED_INTER,
        norm_topk_prob=True,
    )
    ref = qwen3_next_ref.Qwen3NextSparseMoeBlock(args)
    ref.gate.weight = router_dense
    ref.shared_expert.gate_proj.weight = shared_dense["gate_proj"]
    ref.shared_expert.up_proj.weight = shared_dense["up_proj"]
    ref.shared_expert.down_proj.weight = shared_dense["down_proj"]
    ref.shared_expert_gate.weight = shared_gate_dense
    ref.switch_mlp.gate_proj.weight = mx.stack(expert_dense["gate_proj"])
    ref.switch_mlp.up_proj.weight = mx.stack(expert_dense["up_proj"])
    ref.switch_mlp.down_proj.weight = mx.stack(expert_dense["down_proj"])

    yield config, ours, ref
    streamer.close()


def test_moe_block_matches_mlx_lm_reference(moe_pair):
    config, ours, ref = moe_pair
    rng = np.random.default_rng(300)
    x = mx.array(rng.standard_normal(HIDDEN).astype(np.float32))

    indices, weights = ours["router"](x)
    routed = ours["experts"](x, indices, weights)
    got = routed + ours["shared"](x)

    expected = ref(x[None, None])[0, 0]

    mx.eval(got, expected)
    assert mx.allclose(got, expected, rtol=1e-3, atol=1e-4).item()


def test_router_selects_the_same_experts_as_the_reference(moe_pair):
    config, ours, ref = moe_pair
    rng = np.random.default_rng(301)
    x = mx.array(rng.standard_normal(HIDDEN).astype(np.float32))

    indices, scores = ours["router"](x)
    ref_gates = mx.softmax(ref.gate(x[None]), axis=-1, precise=True)
    ref_inds = mx.argpartition(ref_gates, kth=-TOP_K, axis=-1)[..., -TOP_K:]
    ref_scores = mx.take_along_axis(ref_gates, ref_inds, axis=-1)
    ref_scores = ref_scores / ref_scores.sum(axis=-1, keepdims=True)

    mx.eval(indices, scores, ref_inds, ref_scores)
    assert set(indices.tolist()) == set(ref_inds[0].tolist())
    # Scores are order-dependent, so compare them keyed by expert id.
    ours_by_id = dict(zip(indices.tolist(), scores.tolist()))
    ref_by_id = dict(zip(ref_inds[0].tolist(), ref_scores[0].tolist()))
    for expert_id, score in ours_by_id.items():
        assert abs(score - ref_by_id[expert_id]) < 1e-4


def test_norm_topk_prob_changes_the_scores(moe_pair):
    """Confirms the flag is actually wired: with it on the selected scores
    sum to 1, with it off they keep their share of the full softmax."""
    config, ours, _ = moe_pair
    rng = np.random.default_rng(302)
    x = mx.array(rng.standard_normal(HIDDEN).astype(np.float32))

    _, normalized = ours["router"](x)
    unnormalized_router = Qwen3_5Router(
        Qwen3_5Config(
            **{
                **{
                    f.name: getattr(config, f.name)
                    for f in config.__dataclass_fields__.values()
                },
                "norm_topk_prob": False,
            }
        ),  # type: ignore[arg-type]
        ours["router"].gate,
    )
    _, raw = unnormalized_router(x)

    mx.eval(normalized, raw)
    assert abs(float(normalized.sum().item()) - 1.0) < 1e-5
    assert float(raw.sum().item()) < 1.0 - 1e-4


def test_shared_expert_is_gated(moe_pair):
    """The shared expert's contribution is scaled by sigmoid of its own
    gate, so it is never simply the raw MLP output."""
    config, ours, _ = moe_pair
    rng = np.random.default_rng(303)
    x = mx.array(rng.standard_normal(HIDDEN).astype(np.float32))

    gated = ours["shared"](x)
    ungated = swiglu_mlp(x, ours["shared"].projections, config.other_quant)

    mx.eval(gated, ungated)
    assert not mx.allclose(gated, ungated, atol=1e-4).item()


def test_router_order_only_matters_without_norm_topk_prob():
    """Softmax-over-all-then-select and select-then-softmax are identical
    with ``norm_topk_prob`` set, and differ only without it."""
    rng = np.random.default_rng(7)
    logits = mx.array(rng.standard_normal((1, 256)).astype(np.float32) * 2)
    k = 8

    # qwen3_5: softmax over everything, take the top-k, renormalize.
    probabilities = mx.softmax(logits, axis=-1, precise=True)
    qwen_indices = mx.argpartition(-probabilities, kth=k - 1, axis=-1)[..., :k]
    qwen_scores = mx.take_along_axis(probabilities, qwen_indices, axis=-1)
    qwen_normalized = qwen_scores / qwen_scores.sum(axis=-1, keepdims=True)

    # The other order: top-k of the raw logits, softmax over the winners.
    other_indices = mx.argpartition(-logits, kth=k - 1, axis=-1)[..., :k]
    other_scores = mx.softmax(
        mx.take_along_axis(logits, other_indices, axis=-1), axis=-1, precise=True
    )

    mx.eval(qwen_indices, other_indices, qwen_scores, qwen_normalized, other_scores)

    # Monotonicity: the selection cannot differ.
    assert bool((qwen_indices == other_indices).all().item())
    # With renormalization the weights cannot differ either.
    assert float(mx.abs(qwen_normalized - other_scores).max().item()) < 1e-6
    # Without it they genuinely do -- by two orders of magnitude more than the
    # noise above, which is what makes the distinction real for such a model.
    assert float(mx.abs(qwen_scores - other_scores).max().item()) > 1e-2


def test_batched_experts_match_computing_each_position_alone(moe_pair):
    """A wrong inverse permutation still gives every position exactly `k`
    contributions of the right magnitude, just from the wrong experts."""
    _, ours, _ = moe_pair
    rng = np.random.default_rng(400)
    x = mx.array(rng.standard_normal((6, HIDDEN)).astype(np.float32))
    indices, weights = ours["router"](x)

    batched = ours["experts"](x, indices, weights)
    one_at_a_time = mx.stack(
        [ours["experts"](x[t], indices[t], weights[t]) for t in range(x.shape[0])]
    )

    assert batched.shape == one_at_a_time.shape
    assert mx.allclose(batched, one_at_a_time, atol=1e-6), float(
        mx.abs(batched - one_at_a_time).max()
    )


def test_batched_experts_route_each_position_to_its_own_experts(moe_pair):
    """Every position is forced through one expert of its own, so a
    symmetric permutation error cannot hide."""
    _, ours, _ = moe_pair
    rng = np.random.default_rng(401)
    positions = 4
    x = mx.array(rng.standard_normal((positions, HIDDEN)).astype(np.float32))
    # One expert per position, a different one each, full weight on it.
    indices = mx.array([[e] for e in range(positions)])
    weights = mx.ones((positions, 1))

    batched = ours["experts"](x, indices, weights)
    for t in range(positions):
        alone = ours["experts"](x[t], indices[t], weights[t])
        assert mx.allclose(batched[t], alone, atol=1e-6), (
            f"position {t} did not go through expert {t}"
        )
