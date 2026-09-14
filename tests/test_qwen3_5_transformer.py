"""End-to-end `Qwen3_5Transformer` over a tiny 4-layer checkpoint built on
disk, tensor names and all, driven through the real streamer path.

Structural bugs, not numbers: the cross-checks against `mlx_lm` live in
test_qwen3_5_{gated_delta,linear_attention,attention,moe}.py.
"""

from dataclasses import replace

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from mlx_lean_moe.cache.kv_cache import GrowingKVCache
from mlx_lean_moe.cache.recurrent_cache import RecurrentCache
from mlx_lean_moe.config import QuantScheme
from mlx_lean_moe.model.qwen3_5.config import Qwen3_5Config
from mlx_lean_moe.model.qwen3_5.config import Qwen3_5LinearAttention as LinearParams
from mlx_lean_moe.model.qwen3_5.transformer import Qwen3_5Transformer

HIDDEN = 128
GROUP_SIZE = 64
VOCAB = 64
N_HEADS, N_KV_HEADS, HEAD_DIM, ROTARY_DIM = 2, 1, 64, 16
NUM_EXPERTS, TOP_K, MOE_INTER, SHARED_INTER = 4, 2, 64, 128
NUM_LAYERS = 4
PREFIX = "language_model.model"
LINEAR = LinearParams(
    num_key_heads=2,
    num_value_heads=4,
    key_head_dim=64,
    value_head_dim=64,
    conv_kernel_dim=4,
)
# Same shape as the real checkpoint: every layer linear except the last of
# each full_attention_interval group.
IS_LINEAR = (True, True, True, False)


def _quantized(rng, out_dim: int, in_dim: int) -> dict[str, np.ndarray]:
    """Packed-4-bit-affine {weight,scales,biases} with F32 scales; the BF16
    real checkpoints use is covered by test_expert_loader.py."""

    groups, packed = in_dim // GROUP_SIZE, in_dim // 8

    return {
        "weight": rng.integers(0, 2**32, size=(out_dim, packed), dtype=np.uint32),
        "scales": (rng.random((out_dim, groups)).astype(np.float32) * 0.05),
        "biases": (rng.random((out_dim, groups)).astype(np.float32) * 0.05 - 0.025),
    }


def _plain(rng, *shape) -> np.ndarray:
    return ((rng.random(shape).astype(np.float32) * 0.1) + 0.95).astype(np.float32)


def _write_layer(tensors: dict, rng, layer: int, is_linear: bool) -> None:
    prefix = f"{PREFIX}.layers.{layer}"
    for name in ("input_layernorm", "post_attention_layernorm"):
        tensors[f"{prefix}.{name}.weight"] = _plain(rng, HIDDEN)

    if is_linear:
        attn = f"{prefix}.linear_attn"
        for name, out_dim in (
            ("in_proj_qkv", LINEAR.conv_dim),
            ("in_proj_z", LINEAR.value_dim),
            ("in_proj_a", LINEAR.num_value_heads),
            ("in_proj_b", LINEAR.num_value_heads),
        ):
            for field, arr in _quantized(rng, out_dim, HIDDEN).items():
                tensors[f"{attn}.{name}.{field}"] = arr

        for field, arr in _quantized(rng, HIDDEN, LINEAR.value_dim).items():
            tensors[f"{attn}.out_proj.{field}"] = arr

        tensors[f"{attn}.conv1d.weight"] = (
            rng.standard_normal((LINEAR.conv_dim, LINEAR.conv_kernel_dim, 1)) * 0.2
        ).astype(np.float32)
        tensors[f"{attn}.A_log"] = rng.standard_normal(LINEAR.num_value_heads).astype(
            np.float32
        )
        tensors[f"{attn}.dt_bias"] = rng.standard_normal(LINEAR.num_value_heads).astype(
            np.float32
        )
        tensors[f"{attn}.norm.weight"] = _plain(rng, LINEAR.value_head_dim)
    else:
        attn = f"{prefix}.self_attn"
        for name, out_dim, in_dim in (
            (
                "q_proj",
                N_HEADS * HEAD_DIM * 2,
                HIDDEN,
            ),  # queries + the fused output gate
            ("k_proj", N_KV_HEADS * HEAD_DIM, HIDDEN),
            ("v_proj", N_KV_HEADS * HEAD_DIM, HIDDEN),
            ("o_proj", HIDDEN, N_HEADS * HEAD_DIM),
        ):
            for field, arr in _quantized(rng, out_dim, in_dim).items():
                tensors[f"{attn}.{name}.{field}"] = arr

        tensors[f"{attn}.q_norm.weight"] = _plain(rng, HEAD_DIM)
        tensors[f"{attn}.k_norm.weight"] = _plain(rng, HEAD_DIM)

    mlp = f"{prefix}.mlp"
    for field, arr in _quantized(rng, NUM_EXPERTS, HIDDEN).items():
        tensors[f"{mlp}.gate.{field}"] = arr

    for field, arr in _quantized(rng, 1, HIDDEN).items():
        tensors[f"{mlp}.shared_expert_gate.{field}"] = arr

    for name, out_dim, in_dim in (
        ("gate_proj", SHARED_INTER, HIDDEN),
        ("up_proj", SHARED_INTER, HIDDEN),
        ("down_proj", HIDDEN, SHARED_INTER),
    ):
        for field, arr in _quantized(rng, out_dim, in_dim).items():
            tensors[f"{mlp}.shared_expert.{name}.{field}"] = arr

    # Routed experts, stacked along axis 0 per projection.
    for name, out_dim, in_dim in (
        ("gate_proj", MOE_INTER, HIDDEN),
        ("up_proj", MOE_INTER, HIDDEN),
        ("down_proj", HIDDEN, MOE_INTER),
    ):
        stacked = _quantized(rng, out_dim * NUM_EXPERTS, in_dim)
        groups, packed = in_dim // GROUP_SIZE, in_dim // 8

        for field, shape in (
            ("weight", (NUM_EXPERTS, out_dim, packed)),
            ("scales", (NUM_EXPERTS, out_dim, groups)),
            ("biases", (NUM_EXPERTS, out_dim, groups)),
        ):
            tensors[f"{mlp}.switch_mlp.{name}.{field}"] = stacked[field].reshape(shape)


@pytest.fixture
def synthetic_qwen3_5(tmp_path):
    rng = np.random.default_rng(0)
    tensors: dict[str, np.ndarray] = {}

    for layer, is_linear in enumerate(IS_LINEAR):
        _write_layer(tensors, rng, layer, is_linear)

    for field, arr in _quantized(rng, VOCAB, HIDDEN).items():
        tensors[f"{PREFIX}.embed_tokens.{field}"] = arr

    for field, arr in _quantized(rng, VOCAB, HIDDEN).items():
        tensors[f"language_model.lm_head.{field}"] = arr

    tensors[f"{PREFIX}.norm.weight"] = _plain(rng, HIDDEN)
    save_file(tensors, str(tmp_path / "model.safetensors"))

    quant = QuantScheme(bits=4, group_size=GROUP_SIZE)
    config = Qwen3_5Config(
        num_layers=NUM_LAYERS,
        hidden_size=HIDDEN,
        vocab_size=VOCAB,
        rms_norm_eps=1e-6,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        rope_theta=10000000.0,
        rotary_dim=ROTARY_DIM,
        is_linear_per_layer=IS_LINEAR,
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

    return tmp_path, config


def test_transformer_loads_and_decodes(synthetic_qwen3_5):
    import mlx.core as mx

    model_dir, config = synthetic_qwen3_5
    model = Qwen3_5Transformer(model_dir, config, max_context=16)

    try:
        logits = model.prefill([1, 2, 3])
        mx.eval(logits)

        assert logits.shape == (VOCAB,)
        assert not bool(mx.any(mx.isnan(logits)).item())

        step = model(int(mx.argmax(logits).item()))
        mx.eval(step)

        assert step.shape == (VOCAB,)
        assert not bool(mx.any(mx.isnan(step)).item())
    finally:
        model.close()


def test_transformer_prefills_and_decodes_a_mixed_checkpoint(synthetic_qwen3_5):
    import mlx.core as mx

    model_dir, config = synthetic_qwen3_5
    checkpoint = model_dir / "model.safetensors"
    tensors = load_file(checkpoint)
    rng = np.random.default_rng(23)

    # A mixed checkpoint can leave small projections in floating point while
    # packing each routed-expert projection at a different width.
    dense_name = f"{PREFIX}.layers.0.linear_attn.in_proj_a"
    tensors[f"{dense_name}.weight"] = rng.normal(
        0, 0.1, (LINEAR.num_value_heads, HIDDEN)
    ).astype(np.float32)
    del tensors[f"{dense_name}.scales"]
    del tensors[f"{dense_name}.biases"]

    overrides = []

    for projection, bits in (("gate_proj", 3), ("up_proj", 6)):
        name = f"{PREFIX}.layers.0.mlp.switch_mlp.{projection}"
        out_dim = MOE_INTER
        tensors[f"{name}.weight"] = rng.integers(
            0,
            2**32,
            size=(NUM_EXPERTS, out_dim, HIDDEN * bits // 32),
            dtype=np.uint32,
        )
        tensors[f"{name}.scales"] = (
            rng.random((NUM_EXPERTS, out_dim, HIDDEN // 32)).astype(np.float32) * 0.05
        )
        tensors[f"{name}.biases"] = (
            rng.random((NUM_EXPERTS, out_dim, HIDDEN // 32)).astype(np.float32) * 0.05
        )
        overrides.append((name, QuantScheme(bits=bits, group_size=32)))

    save_file(tensors, str(checkpoint))

    mixed_config = replace(config, quant_overrides=tuple(overrides))
    model = Qwen3_5Transformer(model_dir, mixed_config, max_context=16)

    try:
        logits = model.prefill([1, 2, 3])
        step = model(int(mx.argmax(logits).item()))
        mx.eval(logits, step)

        assert logits.shape == step.shape == (VOCAB,)
        assert not bool(mx.any(mx.isnan(logits)).item())
        assert not bool(mx.any(mx.isnan(step)).item())
        assert model.layers[0].experts.projection_quant["gate_proj"].bits == 3
        assert model.layers[0].experts.projection_quant["up_proj"].bits == 6
        assert model.layers[0].attention.in_proj_a.quant is None
    finally:
        model.close()


def test_layers_get_the_cache_kind_their_attention_needs(synthetic_qwen3_5):
    model_dir, config = synthetic_qwen3_5
    model = Qwen3_5Transformer(model_dir, config, max_context=16)

    try:
        kinds = [type(c) for c in model.cache]
        assert kinds == [RecurrentCache, RecurrentCache, RecurrentCache, GrowingKVCache]
    finally:
        model.close()


def test_linear_layer_state_does_not_grow_with_context(synthetic_qwen3_5):
    """The architecture's whole selling point: after prefill *and* several
    decode steps, the linear layers' state is the same size it started."""
    import mlx.core as mx

    model_dir, config = synthetic_qwen3_5
    model = Qwen3_5Transformer(model_dir, config, max_context=32)

    try:
        before = [
            c.ssm_state.shape for c in model.cache if isinstance(c, RecurrentCache)
        ]
        logits = model.prefill([1, 2, 3, 4, 5])

        for _ in range(4):
            logits = model(int(mx.argmax(logits).item()))

        mx.eval(logits)

        after = [
            c.ssm_state.shape for c in model.cache if isinstance(c, RecurrentCache)
        ]

        assert before == after

        # The KV layer, by contrast, really did accumulate all 9 positions.
        kv = [c for c in model.cache if isinstance(c, GrowingKVCache)][0]
        assert kv.size == 9
    finally:
        model.close()


def test_prefill_then_decode_matches_decoding_every_token(synthetic_qwen3_5):
    """Batched prefill and token-by-token feeding must leave both cache kinds
    in the same place, the carried recurrent state especially."""
    import mlx.core as mx

    model_dir, config = synthetic_qwen3_5
    tokens = [1, 2, 3, 4]

    batched = Qwen3_5Transformer(model_dir, config, max_context=16)
    stepwise = Qwen3_5Transformer(model_dir, config, max_context=16)

    try:
        from_prefill = batched.prefill(tokens)

        for t in tokens[:-1]:
            stepwise(t)

        from_steps = stepwise(tokens[-1])
        mx.eval(from_prefill, from_steps)

        assert mx.allclose(from_prefill, from_steps, rtol=1e-3, atol=1e-4).item()
    finally:
        batched.close()

        stepwise.close()


def test_decoder_layer_rejects_3d_input(synthetic_qwen3_5):
    import mlx.core as mx

    model_dir, config = synthetic_qwen3_5
    model = Qwen3_5Transformer(model_dir, config, max_context=16)

    try:
        with pytest.raises(ValueError):
            model.layers[0](mx.zeros((2, 3, HIDDEN)), model.cache[0])
    finally:
        model.close()


@pytest.mark.parametrize("chunk_size", [1, 3, 8])
def test_chunked_prefill_preserves_logits_and_both_cache_states(
    synthetic_qwen3_5, chunk_size
):
    import mlx.core as mx

    model_dir, config = synthetic_qwen3_5
    model = Qwen3_5Transformer(model_dir, config, max_context=32, prefill_chunk_size=32)

    try:
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]

        model.prefill([9, 10])  # check causal masking with an existing prefix

        expected = model.prefill(tokens)
        expected_states = [c.state() for c in model.cache]
        mx.eval(expected, expected_states)

        model.reset_cache()

        model.prefill_chunk_size = chunk_size
        model.prefill([9, 10])

        actual = model.prefill(tokens)
        mx.eval(actual)

        assert mx.allclose(actual, expected, rtol=1e-3, atol=1e-4).item()

        for cache, state in zip(model.cache, expected_states):
            assert cache.size == 10

            for got, want in zip(cache.state(), state):
                assert mx.allclose(got, want, rtol=1e-3, atol=1e-4).item()
    finally:
        model.close()


def test_context_overflow_is_rejected_before_any_layer_changes(synthetic_qwen3_5):
    import mlx.core as mx

    model_dir, config = synthetic_qwen3_5
    model = Qwen3_5Transformer(model_dir, config, max_context=3, prefill_chunk_size=2)

    try:
        mx.eval(model.prefill([1, 2]))

        with pytest.raises(IndexError):
            model.prefill([3, 4])

        assert all(c.size == 2 for c in model.cache)

        mx.eval(model(3))

        with pytest.raises(IndexError):
            model(4)

        assert all(c.size == 3 for c in model.cache)

        with pytest.raises(ValueError):
            model.prefill([])
    finally:
        model.close()


def test_the_default_expert_cache_is_one_slot_per_routed_expert(synthetic_qwen3_5):
    """Measured: on an 8 GB M1 a larger cache bought no throughput and cost
    memory, so the default holds exactly the experts one token routes to."""

    model_dir, config = synthetic_qwen3_5
    model = Qwen3_5Transformer(model_dir, config, max_context=16)

    try:
        assert {loader._max_size for loader in model.expert_loaders} == {TOP_K}
    finally:
        model.close()


def test_an_explicit_expert_cache_overrides_the_default(synthetic_qwen3_5):
    model_dir, config = synthetic_qwen3_5
    model = Qwen3_5Transformer(
        model_dir, config, max_context=16, expert_cache_size_per_layer=TOP_K + 3
    )

    try:
        assert {loader._max_size for loader in model.expert_loaders} == {TOP_K + 3}
    finally:
        model.close()
