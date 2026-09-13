from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import numpy as np
from safetensors.numpy import save_file

from mlx_lean_moe.config import QuantScheme
from mlx_lean_moe.model.moe_block import quantized_linear
from mlx_lean_moe.weights.expert_loader import CachedExpertLoader
from mlx_lean_moe.weights.safetensors_index import build_index
from mlx_lean_moe.weights.stacked_expert_loader import StackedExpertStreamer

NUM_EXPERTS = 4
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
LAYER_PREFIX = "language_model.model.layers"


def _write_stacked_checkpoint(model_dir, num_layers=1):
    """A stacked (num_experts, out, in) tensor per projection/field, the
    real checkpoint's layout, without needing the checkpoint."""
    rng = np.random.default_rng(0)
    all_tensors: dict[str, np.ndarray] = {}
    for layer in range(num_layers):
        for proj in PROJECTIONS:
            prefix = f"{LAYER_PREFIX}.{layer}.experts.switch_glu.{proj}"
            all_tensors[f"{prefix}.weight"] = rng.integers(0, 255, size=(NUM_EXPERTS, 8, 4), dtype=np.uint8)
            all_tensors[f"{prefix}.scales"] = rng.random((NUM_EXPERTS, 8, 1)).astype(np.float32)
            all_tensors[f"{prefix}.biases"] = rng.random((NUM_EXPERTS, 8, 1)).astype(np.float32)
    save_file(all_tensors, str(model_dir / "model.safetensors"))
    return all_tensors


def _streamer_for(model_dir) -> StackedExpertStreamer:
    index = build_index(model_dir, use_cache=False)
    return StackedExpertStreamer(model_dir, index, num_experts=NUM_EXPERTS, layer_prefix=LAYER_PREFIX)


def test_load_expert_matches_an_independently_sliced_reference(tmp_path):
    reference = _write_stacked_checkpoint(tmp_path)
    streamer = _streamer_for(tmp_path)

    loaded = streamer.load_expert(layer=0, expert=2)
    for proj in PROJECTIONS:
        for field in ("weight", "scales", "biases"):
            # Sliced on the reference array, independently of
            # _expert_location's own offset math.
            expected = reference[f"{LAYER_PREFIX}.0.experts.switch_glu.{proj}.{field}"][2]
            actual = np.array(loaded[proj][field])
            np.testing.assert_array_equal(actual, expected)

    streamer.close()


def test_load_expert_only_reads_the_requested_expert(tmp_path):
    reference = _write_stacked_checkpoint(tmp_path)
    streamer = _streamer_for(tmp_path)

    loaded = streamer.load_expert(layer=0, expert=0)
    other_expert_weight = reference[f"{LAYER_PREFIX}.0.experts.switch_glu.gate_proj.weight"][1]
    this_expert_weight = np.array(loaded["gate_proj"]["weight"])
    assert this_expert_weight.shape == other_expert_weight.shape
    assert not np.array_equal(this_expert_weight, other_expert_weight)

    streamer.close()


def test_load_experts_concurrently_matches_sequential_load_expert(tmp_path):
    _write_stacked_checkpoint(tmp_path)
    streamer = _streamer_for(tmp_path)

    expected = [streamer.load_expert(0, e) for e in (2, 0, 3, 1)]
    with ThreadPoolExecutor(max_workers=9) as executor:
        actual = streamer.load_experts_concurrently(0, [2, 0, 3, 1], executor)

    for e, a in zip(expected, actual):
        for proj in PROJECTIONS:
            for field in ("weight", "scales", "biases"):
                np.testing.assert_array_equal(np.array(e[proj][field]), np.array(a[proj][field]))

    streamer.close()


def test_cached_expert_loader_wraps_stacked_streamer_with_no_changes(tmp_path):
    """CachedExpertLoader is duck-typed on
    load_expert/load_experts_concurrently, so this needs no changes."""
    _write_stacked_checkpoint(tmp_path)
    streamer = _streamer_for(tmp_path)
    cache = CachedExpertLoader(streamer, max_size=2)

    cache.load_expert(0, 0)
    cache.load_expert(0, 0)
    cache.load_expert(0, 1)
    assert cache.hits == 1
    assert cache.misses == 2
    assert (0, 0) in cache and (0, 1) in cache

    # Which entry goes is the eviction policy's business, pinned in
    # test_expert_loader.py; here it is only that the cache stays bounded.
    cache.load_expert(0, 2)
    assert len(cache) == 2

    streamer.close()


def test_stacked_experts_mix_3bit_6bit_and_dense_projections(tmp_path):
    rng = np.random.default_rng(19)
    schemes = {
        "gate_proj": QuantScheme(bits=3, group_size=32),
        "down_proj": QuantScheme(bits=6, group_size=32),
    }
    tensors: dict[str, np.ndarray] = {}
    references: dict[str, list[np.ndarray]] = {proj: [] for proj in PROJECTIONS}
    for proj in PROJECTIONS:
        per_field: dict[str, list[np.ndarray]] = {}
        for _ in range(NUM_EXPERTS):
            weight = rng.normal(0, 0.1, (8, 64)).astype(np.float32)
            references[proj].append(weight)
            if proj == "up_proj":
                per_field.setdefault("weight", []).append(weight)
            else:
                scheme = schemes[proj]
                packed, scales, biases = mx.quantize(mx.array(weight), group_size=scheme.group_size, bits=scheme.bits)
                mx.eval(packed, scales, biases)
                per_field.setdefault("weight", []).append(np.array(packed))
                per_field.setdefault("scales", []).append(np.array(scales))
                per_field.setdefault("biases", []).append(np.array(biases))
        prefix = f"{LAYER_PREFIX}.0.experts.switch_glu.{proj}"
        for field, values in per_field.items():
            tensors[f"{prefix}.{field}"] = np.stack(values)
    save_file(tensors, str(tmp_path / "model.safetensors"))

    streamer = _streamer_for(tmp_path)
    sequential = streamer.load_expert(0, 2)
    with ThreadPoolExecutor(max_workers=7) as executor:
        concurrent = streamer.load_experts_concurrently(0, [2], executor)[0]

    assert isinstance(sequential["up_proj"], mx.array)
    np.testing.assert_array_equal(np.array(sequential["up_proj"]), references["up_proj"][2])
    for proj in ("gate_proj", "down_proj"):
        assert set(sequential[proj]) == {"weight", "scales", "biases"}
        for field in sequential[proj]:
            np.testing.assert_array_equal(np.array(sequential[proj][field]), np.array(concurrent[proj][field]))

    x = mx.array(rng.normal(0, 0.1, (2, 64)).astype(np.float32))
    # Dense expert projections ignore the global quantization fallback and
    # execute as an ordinary matrix multiplication.
    dense_actual = quantized_linear(x, sequential["up_proj"], QuantScheme(bits=4, group_size=64))
    np.testing.assert_allclose(np.array(dense_actual), np.array(x @ mx.array(references["up_proj"][2]).T))
    for proj in ("gate_proj", "down_proj"):
        actual = quantized_linear(x, sequential[proj], schemes[proj])
        reference = (
            x
            @ mx.dequantize(
                sequential[proj]["weight"],
                scales=sequential[proj]["scales"],
                biases=sequential[proj]["biases"],
                group_size=schemes[proj].group_size,
                bits=schemes[proj].bits,
            ).T
        )
        np.testing.assert_allclose(np.array(actual), np.array(reference), rtol=2e-4, atol=2e-4)
    streamer.close()
