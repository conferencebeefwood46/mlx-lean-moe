import json
import struct

import numpy as np
import pytest
from safetensors.numpy import save_file

from mlx_lean_moe.weights.safetensors_index import build_index, checkpoint_is_complete


def _write_two_shard_checkpoint(model_dir):
    shard0 = {
        "embed.weight": np.arange(24, dtype=np.float32).reshape(4, 6),
        "layers.0.gate.weight": np.arange(12, dtype=np.uint8).reshape(3, 4),
    }
    shard1 = {
        "layers.0.expert.3.weight": np.arange(20, dtype=np.float32).reshape(4, 5),
        "layers.0.expert.7.weight": np.arange(30, dtype=np.float32).reshape(5, 6),
    }
    save_file(shard0, str(model_dir / "model-00001-of-00002.safetensors"))
    save_file(shard1, str(model_dir / "model-00002-of-00002.safetensors"))

    weight_map = {name: "model-00001-of-00002.safetensors" for name in shard0}
    weight_map.update({name: "model-00002-of-00002.safetensors" for name in shard1})
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map})
    )
    return shard0 | shard1


def test_index_covers_every_tensor_across_shards(tmp_path):
    tensors = _write_two_shard_checkpoint(tmp_path)
    index = build_index(tmp_path, use_cache=False)
    assert set(index) == set(tensors)


def test_index_offsets_match_raw_bytes(tmp_path):
    tensors = _write_two_shard_checkpoint(tmp_path)
    index = build_index(tmp_path, use_cache=False)

    for name, expected in tensors.items():
        loc = index[name]
        assert loc.shape == expected.shape
        with (tmp_path / loc.shard).open("rb") as f:
            f.seek(loc.offset)
            raw = f.read(loc.length)
        actual = np.frombuffer(raw, dtype=expected.dtype).reshape(expected.shape)
        np.testing.assert_array_equal(actual, expected)


def test_sidecar_cache_is_written_and_reused(tmp_path, monkeypatch):
    _write_two_shard_checkpoint(tmp_path)
    first = build_index(tmp_path, use_cache=True)
    sidecar = tmp_path / ".mlx_lean_moe_index.json"
    assert sidecar.exists()

    # Corrupt the shard headers so a real re-parse would blow up, proving the
    # second call actually took the cached path instead of re-reading them.
    for shard in tmp_path.glob("*.safetensors"):
        with shard.open("r+b") as f:
            f.seek(0)
            f.write(struct.pack("<Q", 999_999))

    second = build_index(tmp_path, use_cache=True)
    assert second == first


def test_single_shard_checkpoint_without_index_json(tmp_path):
    save_file({"a": np.zeros(3, dtype=np.float32)}, str(tmp_path / "model.safetensors"))
    index = build_index(tmp_path, use_cache=False)
    assert set(index) == {"a"}
    assert index["a"].shard == "model.safetensors"


def test_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_index(tmp_path, use_cache=False)


def test_complete_checkpoint_is_recognized_without_the_network(tmp_path):
    _write_two_shard_checkpoint(tmp_path)
    assert checkpoint_is_complete(tmp_path) is True


def test_a_truncated_shard_is_not_complete(tmp_path):
    """A shard's header says where its data ends, so a truncated file is
    detectable even though its shapes and offsets parse fine."""
    _write_two_shard_checkpoint(tmp_path)
    shard = tmp_path / "model-00002-of-00002.safetensors"
    whole = shard.stat().st_size
    with shard.open("r+b") as f:
        f.truncate(whole - 4)  # one float short

    assert checkpoint_is_complete(tmp_path) is False


def test_a_shard_the_index_names_but_that_is_absent_is_not_complete(tmp_path):
    _write_two_shard_checkpoint(tmp_path)
    (tmp_path / "model-00002-of-00002.safetensors").unlink()
    assert checkpoint_is_complete(tmp_path) is False


def test_a_shard_truncated_inside_its_own_header_is_not_complete(tmp_path):
    """Truncated so early that the header JSON itself is cut off -- must
    answer False rather than raise out of a completeness check."""
    _write_two_shard_checkpoint(tmp_path)
    shard = tmp_path / "model-00001-of-00002.safetensors"
    with shard.open("r+b") as f:
        f.truncate(12)  # 8-byte length prefix plus 4 bytes of JSON
    assert checkpoint_is_complete(tmp_path) is False


def test_an_empty_directory_is_not_complete(tmp_path):
    assert checkpoint_is_complete(tmp_path) is False
