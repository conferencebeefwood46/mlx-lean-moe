import json
import time
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import numpy as np
import pytest
from safetensors.numpy import save_file

from mlx_lean_moe.weights.expert_loader import (
    CachedExpertLoader,
    TensorStreamer,
    decode_tensor,
)
from mlx_lean_moe.weights.safetensors_index import build_index
from mlx_lean_moe.weights.stacked_expert_loader import StackedExpertStreamer

NUM_EXPERTS = 4
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


LAYER_PREFIX = "language_model.model.layers"


def _write_checkpoint(model_dir, num_layers=1):
    """The stacked layout real checkpoints use: one tensor per
    projection/field with the experts along axis 0."""
    rng = np.random.default_rng(0)
    all_tensors: dict[str, np.ndarray] = {}
    for layer in range(num_layers):
        for proj in PROJECTIONS:
            prefix = f"{LAYER_PREFIX}.{layer}.experts.switch_glu.{proj}"
            all_tensors[f"{prefix}.weight"] = rng.integers(
                0, 255, size=(NUM_EXPERTS, 8, 4), dtype=np.uint8
            )
            all_tensors[f"{prefix}.scales"] = rng.random((NUM_EXPERTS, 8, 1)).astype(
                np.float32
            )
            all_tensors[f"{prefix}.biases"] = rng.random((NUM_EXPERTS, 8, 1)).astype(
                np.float32
            )
    save_file(all_tensors, str(model_dir / "model.safetensors"))
    return all_tensors


def _streamer_for(model_dir) -> StackedExpertStreamer:
    index = build_index(model_dir, use_cache=False)
    return StackedExpertStreamer(
        model_dir, index, num_experts=NUM_EXPERTS, layer_prefix=LAYER_PREFIX
    )


def _write_raw_safetensors(
    path, entries: dict[str, tuple[str, tuple[int, ...], bytes]]
) -> None:
    """A minimal safetensors file for dtypes `safetensors.numpy` cannot
    write itself, namely BF16."""
    import struct

    header = {}
    data = bytearray()
    for name, (dtype, shape, raw) in entries.items():
        start = len(data)
        data += raw
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [start, len(data)],
        }
    header_bytes = json.dumps(header).encode()
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(bytes(data))


def _placeholder(dtype="U8", shape=(1, 1)) -> tuple[str, tuple[int, ...], bytes]:
    return dtype, shape, bytes(1)


def _bf16_bytes(values: np.ndarray) -> bytes:
    return (values.view(np.uint32) >> 16).astype(np.uint16).tobytes()


def test_bf16_widens_for_arithmetic_and_stays_narrow_for_metadata():
    """Tensors feeding float32 arithmetic widen; scales and biases, consumed
    bit-identically at their stored width, stay narrow."""
    values = np.array([1.5, -2.25, 3.0, 0.5, 0.0], dtype=np.float32)

    widened = decode_tensor(_bf16_bytes(values), "BF16", (values.size,))
    assert widened.dtype == mx.float32
    np.testing.assert_array_equal(np.array(widened), values)

    kept = decode_tensor(_bf16_bytes(values), "BF16", (values.size,), stored_width=True)
    assert kept.dtype == mx.bfloat16
    np.testing.assert_array_equal(np.array(kept.astype(mx.float32)), values)


def test_stored_width_follows_the_field_name_through_a_real_file(tmp_path):
    """The same through the streamer, so the header's dtype and the byte
    offsets are exercised too."""
    values = np.array([[1.5, -2.25, 3.0, 0.5]], dtype=np.float32)
    raw = _bf16_bytes(values)
    _write_raw_safetensors(
        tmp_path / "model.safetensors",
        {
            "layer.scales": ("BF16", values.shape, raw),
            "layer.biases": ("BF16", values.shape, raw),
            # Not metadata, despite living beside it and being the same dtype.
            "layer.norm.weight": ("BF16", values.shape, raw),
            "other": _placeholder(),
        },
    )

    streamer = TensorStreamer(tmp_path, build_index(tmp_path, use_cache=False))
    for name in ("layer.scales", "layer.biases"):
        tensor = streamer.read_tensor(name)
        assert tensor.dtype == mx.bfloat16, name
        np.testing.assert_array_equal(np.array(tensor.astype(mx.float32)), values)

    weight = streamer.read_tensor("layer.norm.weight")
    assert weight.dtype == mx.float32
    np.testing.assert_array_equal(np.array(weight), values)
    streamer.close()


def test_cached_loader_only_misses_on_first_access(tmp_path):
    _write_checkpoint(tmp_path)
    streamer = _streamer_for(tmp_path)
    cache = CachedExpertLoader(streamer, max_size=2)

    cache.load_expert(0, 0)
    cache.load_expert(0, 0)
    cache.load_expert(0, 0)

    assert cache.misses == 1
    assert cache.hits == 2
    streamer.close()


def test_cached_loader_rejects_nonpositive_max_size(tmp_path):
    _write_checkpoint(tmp_path)
    streamer = _streamer_for(tmp_path)
    with pytest.raises(ValueError):
        CachedExpertLoader(streamer, max_size=0)
    streamer.close()


def test_eviction_is_by_recency_not_frequency(tmp_path):
    """(0,0) is used twice but not recently; (0,1) once but more recently.
    Frequency would keep (0,0); recency keeps (0,1)."""
    _write_checkpoint(tmp_path)
    streamer = _streamer_for(tmp_path)
    cache = CachedExpertLoader(streamer, max_size=2)

    cache.load_expert(0, 0)
    cache.load_expert(0, 0)  # used twice, but now the older of the two
    cache.load_expert(0, 1)  # used once, and the most recent
    cache.load_expert(0, 2)  # forces eviction

    assert (0, 0) not in cache  # frequent, but least recently used
    assert (0, 1) in cache
    streamer.close()


def test_a_hit_renews_an_entry(tmp_path):
    """A hit has to move the key to the back of the queue, or the cache is
    a FIFO wearing the name."""
    _write_checkpoint(tmp_path)
    streamer = _streamer_for(tmp_path)
    cache = CachedExpertLoader(streamer, max_size=2)

    cache.load_expert(0, 0)
    cache.load_expert(0, 1)
    cache.load_expert(0, 0)  # a hit, which renews (0,0) and leaves (0,1) oldest
    cache.load_expert(0, 2)

    assert (0, 1) not in cache
    assert (0, 0) in cache
    assert cache.hits == 1
    streamer.close()


def test_load_many_classifies_a_whole_batch_before_inserting(tmp_path):
    """A batch is classified against the cache as it stood when the batch
    arrived: an insert mid-batch could evict a key later in the same batch."""
    _write_checkpoint(tmp_path)
    streamer = _streamer_for(tmp_path)
    cache = CachedExpertLoader(streamer, max_size=2)

    cache.load_many(0, [0, 1])
    assert (cache.hits, cache.misses) == (0, 2)

    # (0,2) misses and, being inserted, must evict something -- but (0,0) and
    # (0,1) were both resident when the batch arrived, so both count as hits.
    cache.load_many(0, [0, 2, 1])
    assert (cache.hits, cache.misses) == (2, 3)
    assert len(cache) == 2
    streamer.close()


def test_load_many_matches_individually_loading_each_expert(tmp_path):
    _write_checkpoint(tmp_path)
    streamer = _streamer_for(tmp_path)

    individually = CachedExpertLoader(streamer, max_size=NUM_EXPERTS)
    expected = [individually.load_expert(0, e) for e in (2, 0, 3, 1)]

    with ThreadPoolExecutor(max_workers=4) as executor:
        batched = CachedExpertLoader(streamer, max_size=NUM_EXPERTS, executor=executor)
        actual = batched.load_many(0, [2, 0, 3, 1])

    for e, a in zip(expected, actual):
        for proj in PROJECTIONS:
            for field in ("weight", "scales", "biases"):
                np.testing.assert_array_equal(
                    np.array(e[proj][field]), np.array(a[proj][field])
                )

    streamer.close()


def test_load_many_mixes_hits_and_misses_correctly(tmp_path):
    _write_checkpoint(tmp_path)
    streamer = _streamer_for(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as executor:
        cache = CachedExpertLoader(streamer, max_size=NUM_EXPERTS, executor=executor)

        cache.load_expert(0, 0)  # warm one entry
        results = cache.load_many(0, [0, 1, 2])  # 0 is a hit, 1 and 2 are misses

        assert cache.hits == 1
        assert (
            cache.misses == 3
        )  # the initial load_expert(0,0) + the 2 misses in load_many
        assert len(results) == 3
        assert (0, 0) in cache and (0, 1) in cache and (0, 2) in cache
    streamer.close()


def test_load_many_runs_cache_misses_concurrently(tmp_path):
    """A slow (artificially delayed) streamer proves the misses actually
    overlap in wall time instead of running one after another."""
    _write_checkpoint(tmp_path)
    real_streamer = _streamer_for(tmp_path)

    class SlowStreamer:
        def load_expert(self, layer, expert):
            time.sleep(0.1)
            return real_streamer.load_expert(layer, expert)

        def load_experts_concurrently(self, layer, experts, executor):
            futures = [executor.submit(self.load_expert, layer, e) for e in experts]
            return [f.result() for f in futures]

    slow = SlowStreamer()
    k = 4
    with ThreadPoolExecutor(max_workers=k) as executor:
        cache = CachedExpertLoader(slow, max_size=k, executor=executor)
        t0 = time.time()
        cache.load_many(0, [0, 1, 2, 3])
        elapsed = time.time() - t0

    # k=4 misses at 0.1s each: ~0.1s if concurrent, ~0.4s if sequential.
    assert elapsed < 0.3, f"expected concurrent misses to overlap, took {elapsed:.2f}s"
    real_streamer.close()


def test_load_many_without_an_executor_still_works_sequentially(tmp_path):
    _write_checkpoint(tmp_path)
    streamer = _streamer_for(tmp_path)
    cache = CachedExpertLoader(streamer, max_size=NUM_EXPERTS)  # no executor
    results = cache.load_many(0, [0, 1, 2])
    assert len(results) == 3
    assert cache.misses == 3
    streamer.close()


def test_decode_tensor_is_safe_off_the_main_thread():
    """What `decode_tensor` returns must survive evaluation on another
    thread: MLX's default stream is thread-local."""
    from concurrent.futures import ThreadPoolExecutor

    cases = [
        (np.array([1.5, -2.25], dtype=np.float32).tobytes(), "F32", (2,), False),
        (np.array([0x3F80, 0xC000], dtype=np.uint16).tobytes(), "BF16", (2,), False),
        # The metadata path really does build an op -- a reinterpreting view --
        # so it is the one that has to name the stream it belongs to.
        (np.array([0x3F80, 0xC000], dtype=np.uint16).tobytes(), "BF16", (2,), True),
        (np.array([7, 9], dtype=np.uint32).tobytes(), "U32", (2,), False),
    ]

    def decode(case):
        raw, dtype, shape, stored_width = case
        return decode_tensor(raw, dtype, shape, stored_width=stored_width)

    # Several workers, so the decoding genuinely happens off this thread.
    with ThreadPoolExecutor(max_workers=4) as executor:
        arrays = list(executor.map(decode, cases * 4))

    # The failure is at evaluation, on this thread -- not where it was built.
    mx.eval(arrays)
    assert float(arrays[1][0].item()) == 1.0
    assert arrays[2].dtype == mx.bfloat16
    assert float(arrays[2][0].item()) == 1.0
