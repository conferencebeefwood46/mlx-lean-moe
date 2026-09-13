import mlx.core as mx
import numpy as np
import pytest
from safetensors.numpy import save_file

from mlx_lean_moe.config import QuantScheme
from mlx_lean_moe.model.moe_block import LinearWeights, quantized_linear
from mlx_lean_moe.weights.expert_loader import TensorStreamer, read_linear
from mlx_lean_moe.weights.safetensors_index import build_index


def _quantize(weight: np.ndarray, scheme: QuantScheme) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    packed, scales, biases = mx.quantize(
        mx.array(weight),
        group_size=scheme.group_size,
        bits=scheme.bits,
        mode=scheme.mode,
    )
    mx.eval(packed, scales, biases)
    return np.array(packed), np.array(scales), np.array(biases)


def _add_quantized(tensors: dict[str, np.ndarray], name: str, weight: np.ndarray, scheme: QuantScheme) -> None:
    packed, scales, biases = _quantize(weight, scheme)
    tensors[f"{name}.weight"] = packed
    tensors[f"{name}.scales"] = scales
    tensors[f"{name}.biases"] = biases


def test_linear_loader_runs_adjacent_projections_with_their_own_schemes(tmp_path):
    rng = np.random.default_rng(7)
    schemes = {
        "two_bit_proj": QuantScheme(bits=2, group_size=64),
        "gate_proj": QuantScheme(bits=3, group_size=32),
        "up_proj": QuantScheme(bits=4, group_size=64),
        "down_proj": QuantScheme(bits=6, group_size=32),
        "eight_bit_proj": QuantScheme(bits=8, group_size=64),
    }
    dense_weight = rng.normal(0, 0.1, (9, 64)).astype(np.float32)
    source_weights = {name: rng.normal(0, 0.1, (9, 64)).astype(np.float32) for name in schemes}
    tensors: dict[str, np.ndarray] = {"dense_proj.weight": dense_weight}
    for name, scheme in schemes.items():
        _add_quantized(tensors, name, source_weights[name], scheme)
    save_file(tensors, str(tmp_path / "model.safetensors"))

    streamer = TensorStreamer(tmp_path, build_index(tmp_path, use_cache=False))
    x = mx.array(rng.normal(0, 0.1, (3, 64)).astype(np.float32))
    for name, scheme in schemes.items():
        loaded = read_linear(streamer, name, scheme)
        assert isinstance(loaded, LinearWeights)
        assert loaded.quant == scheme
        actual = quantized_linear(x, loaded)
        reference = (
            x
            @ mx.dequantize(
                loaded.tensors["weight"],
                scales=loaded.tensors["scales"],
                biases=loaded.tensors["biases"],
                group_size=scheme.group_size,
                bits=scheme.bits,
            ).T
        )
        np.testing.assert_allclose(np.array(actual), np.array(reference), rtol=2e-4, atol=2e-4)

    with pytest.raises(ValueError, match="expected packed width"):
        read_linear(streamer, "gate_proj", QuantScheme(bits=4, group_size=32))

    dense = read_linear(streamer, "dense_proj", QuantScheme(bits=4, group_size=64))
    assert dense.quant is None
    np.testing.assert_allclose(np.array(quantized_linear(x, dense)), np.array(x @ mx.array(dense_weight).T))
    streamer.close()


def test_linear_row_streaming_supports_quantized_and_dense_tables(tmp_path):
    rng = np.random.default_rng(11)
    scheme = QuantScheme(bits=3, group_size=32)
    quantized_weight = rng.normal(0, 0.1, (7, 64)).astype(np.float32)
    dense_weight = rng.normal(0, 0.1, (7, 64)).astype(np.float32)
    tensors = {"dense.weight": dense_weight}
    _add_quantized(tensors, "quantized", quantized_weight, scheme)
    save_file(tensors, str(tmp_path / "model.safetensors"))

    streamer = TensorStreamer(tmp_path, build_index(tmp_path, use_cache=False))
    rows = [5, 1, 5]
    actual = streamer.read_linear_rows("quantized", rows, scheme)
    loaded = read_linear(streamer, "quantized", scheme).tensors
    reference = mx.dequantize(
        loaded["weight"][rows],
        scales=loaded["scales"][rows],
        biases=loaded["biases"][rows],
        group_size=scheme.group_size,
        bits=scheme.bits,
    )
    np.testing.assert_allclose(np.array(actual), np.array(reference))
    np.testing.assert_array_equal(
        np.array(streamer.read_linear_rows("dense", rows, scheme)),
        dense_weight[rows],
    )
    streamer.close()


def test_linear_loader_rejects_partial_quantization_fields(tmp_path):
    save_file(
        {
            "broken.weight": np.zeros((2, 2), dtype=np.float32),
            "broken.biases": np.zeros((2, 1), dtype=np.float32),
        },
        str(tmp_path / "model.safetensors"),
    )
    streamer = TensorStreamer(tmp_path, build_index(tmp_path, use_cache=False))
    with pytest.raises(ValueError, match="biases but no scales"):
        read_linear(streamer, "broken", QuantScheme(bits=4, group_size=64))
    streamer.close()


@pytest.mark.parametrize(
    "scheme",
    [
        (7, 64, "affine"),
        (8, 32, "mxfp4"),
        (4, 64, "mxfp4"),
        (4, 32, "unknown"),
    ],
)
def test_quant_scheme_rejects_combinations_mlx_cannot_execute(scheme):
    with pytest.raises(ValueError, match="unsupported"):
        QuantScheme(*scheme)
