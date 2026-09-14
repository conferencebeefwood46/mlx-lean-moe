"""Repacking moves an expert's bytes without changing them.

Every test compares against the stacked streamer reading the same
checkpoint, so the two implementations have to agree.
"""

import json
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import numpy as np
import pytest
from safetensors.numpy import save_file

from mlx_lean_moe.weights.expert_pack import (
    INDEX_NAME,
    PACK_NAME,
    ExpertPackStreamer,
    build,
    checkpoint_dir,
    load_layout,
    verify,
)
from mlx_lean_moe.weights.safetensors_index import build_index
from mlx_lean_moe.weights.stacked_expert_loader import StackedExpertStreamer

LAYER_PREFIX = "language_model.model.layers"
STACK = "experts.switch_glu"
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
NUM_EXPERTS = 6
NUM_LAYERS = 2


def _write_checkpoint(model_dir, dense: tuple[str, ...] = ()) -> dict[str, np.ndarray]:
    """Interleaves the fields of different projections so a pack that assumed
    a tidy on-disk order would be caught."""
    rng = np.random.default_rng(3)
    tensors: dict[str, np.ndarray] = {}
    for layer in range(NUM_LAYERS):
        for projection in PROJECTIONS:
            prefix = f"{LAYER_PREFIX}.{layer}.{STACK}.{projection}"
            tensors[f"{prefix}.weight"] = rng.integers(
                0, 2**31, size=(NUM_EXPERTS, 4, 3), dtype=np.uint32
            )
            if projection in dense:
                continue
            tensors[f"{prefix}.scales"] = rng.random((NUM_EXPERTS, 4, 2)).astype(
                np.float32
            )
            tensors[f"{prefix}.biases"] = rng.random((NUM_EXPERTS, 4, 2)).astype(
                np.float32
            )
    save_file(tensors, str(model_dir / "model.safetensors"))
    return tensors


def _streamers(model_dir):
    index = build_index(model_dir, use_cache=False)
    original = StackedExpertStreamer(
        model_dir,
        index,
        num_experts=NUM_EXPERTS,
        layer_prefix=LAYER_PREFIX,
        stack_name=STACK,
    )
    build(
        model_dir,
        index,
        num_experts=NUM_EXPERTS,
        num_layers=NUM_LAYERS,
        layer_prefix=LAYER_PREFIX,
        stack_name=STACK,
    )
    layout = load_layout(model_dir)
    assert layout is not None
    return original, ExpertPackStreamer(model_dir, layout)


def test_a_packed_expert_is_the_checkpoint_s_expert(tmp_path):
    _write_checkpoint(tmp_path)
    original, packed = _streamers(tmp_path)
    try:
        verify(packed, original, layers=range(NUM_LAYERS), experts=range(NUM_EXPERTS))
    finally:
        original.close()
        packed.close()


def test_a_dense_projection_survives_the_round_trip(tmp_path):
    """A mixed checkpoint can leave one projection dense, so the pack has to
    record what each layer has rather than assuming nine fields."""
    _write_checkpoint(tmp_path, dense=("up_proj",))
    original, packed = _streamers(tmp_path)
    try:
        one = packed.load_expert(0, 2)
        assert isinstance(one["up_proj"], mx.array)
        assert set(one["gate_proj"]) == {"weight", "scales", "biases"}
        verify(packed, original, layers=range(NUM_LAYERS), experts=range(NUM_EXPERTS))
    finally:
        original.close()
        packed.close()


def test_concurrent_reads_match_sequential_ones(tmp_path):
    _write_checkpoint(tmp_path)
    original, packed = _streamers(tmp_path)
    try:
        wanted = [4, 0, 5, 1]
        expected = [packed.load_expert(1, expert) for expert in wanted]
        with ThreadPoolExecutor(max_workers=4) as executor:
            actual = packed.load_experts_concurrently(1, wanted, executor)
        # Order asked for, not order finished in.
        for mine, reference in zip(actual, expected):
            for projection, tensors in reference.items():
                for field, tensor in tensors.items():
                    assert mx.array_equal(mine[projection][field], tensor).item()
    finally:
        original.close()
        packed.close()


def test_one_expert_is_one_read(tmp_path):
    """The point of the pack: nine scattered ranges become one, and read
    count is what the wait follows."""
    _write_checkpoint(tmp_path)
    original, packed = _streamers(tmp_path)
    try:
        layout = packed.layout
        offset, stride = layout.expert_range(0, 3)
        assert stride == sum(f["length"] for f in layout.fields(0))
        # Consecutive experts are adjacent, so a whole batch is one run too.
        assert layout.expert_range(0, 4)[0] == offset + stride
    finally:
        original.close()
        packed.close()


def test_a_pack_from_another_version_is_refused(tmp_path):
    _write_checkpoint(tmp_path)
    original, packed = _streamers(tmp_path)
    original.close()
    packed.close()

    document = json.loads((tmp_path / INDEX_NAME).read_text())
    document["version"] = 99
    (tmp_path / INDEX_NAME).write_text(json.dumps(document))
    with pytest.raises(ValueError, match="version"):
        load_layout(tmp_path)


def test_no_pack_means_no_layout(tmp_path):
    _write_checkpoint(tmp_path)
    assert load_layout(tmp_path) is None
    original, packed = _streamers(tmp_path)
    original.close()
    packed.close()
    (tmp_path / PACK_NAME).unlink()
    # The index alone is not a pack: deleting the data must not leave the
    # engine thinking it can read from it.
    assert load_layout(tmp_path) is None


def test_a_checkpoint_directory_resolves_to_itself(tmp_path):
    assert checkpoint_dir(tmp_path) == tmp_path
    assert checkpoint_dir(str(tmp_path)) == tmp_path


def test_a_repo_id_resolves_through_the_cache(tmp_path, monkeypatch):
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    monkeypatch.setattr(
        "mlx_lean_moe.weights.download.snapshot_dir",
        lambda repo_id, *a, **k: snapshot if repo_id == "owner/name" else None,
    )
    assert checkpoint_dir("owner/name") == snapshot


def test_an_undownloaded_repo_id_says_how_to_fetch_it(monkeypatch):
    """Rather than a bare "no such file", which reads as a bug in the path
    the user typed."""
    monkeypatch.setattr(
        "mlx_lean_moe.weights.download.snapshot_dir", lambda *a, **k: None
    )
    with pytest.raises(
        FileNotFoundError, match="mlx_lean_moe.weights.download owner/name"
    ):
        checkpoint_dir("owner/name")


def test_a_missing_directory_is_not_retried_as_a_repo_id(tmp_path, monkeypatch):
    """A repo id has exactly the shape of a relative path, so a mistyped
    path must not come back as a confusing message about the hub."""
    monkeypatch.setattr(
        "mlx_lean_moe.weights.download.snapshot_dir",
        lambda *a, **k: pytest.fail(
            "a filesystem path must not be looked up on the hub"
        ),
    )
    with pytest.raises((NotADirectoryError, FileNotFoundError)):
        checkpoint_dir(tmp_path / "absent")


def test_a_file_is_not_a_checkpoint_directory(tmp_path):
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"")
    with pytest.raises(NotADirectoryError):
        checkpoint_dir(weights)
