"""`GrowingKVCache`: the key/value history for a softmax-attention layer."""

import gc

import mlx.core as mx
import pytest

from mlx_lean_moe.cache.kv_cache import GrowingKVCache

NUM_KV_HEADS = 2
HEAD_DIM = 3


def _token(value: float) -> mx.array:
    """A key/value tensor for one token, filled with a single distinguishing
    constant so equality checks reduce to comparing scalars."""

    return mx.full((NUM_KV_HEADS, HEAD_DIM), value, dtype=mx.float16)


def _append_tokens(cache, values: list[float]) -> None:
    for v in values:
        cache.append(_token(v), _token(v))


def _state_values(cache) -> list[float]:
    keys, _ = cache.state()  # (num_kv_heads, n_tokens, head_dim)
    return [float(keys[0, i, 0].item()) for i in range(keys.shape[1])]


def test_growing_cache_rejects_nonpositive_context():
    with pytest.raises(ValueError):
        GrowingKVCache(max_context=0, num_kv_heads=1, head_dim=1)


def test_growing_cache_matches_full_history():
    cache = GrowingKVCache(max_context=16, num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM)
    values = [float(i) for i in range(7)]
    _append_tokens(cache, values)

    assert cache.size == 7
    assert _state_values(cache) == values


def test_growing_cache_raises_once_context_is_exceeded():
    """Wrapping instead would be silent: the layer would go on attending to a
    history missing its oldest tokens, fluently and wrongly."""

    cache = GrowingKVCache(max_context=3, num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM)
    _append_tokens(cache, [1.0, 2.0, 3.0])

    with pytest.raises(IndexError):
        cache.append(_token(4.0), _token(4.0))


def test_device_memory_tracks_the_history_and_nothing_else():
    """Buffer updates are lazy, so shapes stay right either way; bytes can
    tell. Both readings follow a `gc.collect()`, being process-wide."""

    heads, dim, context = 4, 256, 2200
    cache = GrowingKVCache(
        max_context=context, num_kv_heads=heads, head_dim=dim, dtype=mx.float16
    )

    def decode_step(i: int) -> None:
        token = mx.full((heads, dim), float(i), dtype=mx.float16)
        cache.append(token, token)
        mx.eval(cache._keys, cache._values)  # what the decode loop does anyway

    for i in range(50):  # let the steady-state buffers settle
        decode_step(i)

    gc.collect()

    baseline = mx.get_active_memory()

    for i in range(50, 2000):
        decode_step(i)

    gc.collect()

    grown = mx.get_active_memory()
    assert grown <= baseline + 8 * 2**20, (
        f"device memory grew from {baseline / 2**20:.1f} MB to {grown / 2**20:.1f} MB "
        f"over 1950 decode steps; only the growing history should remain"
    )


def test_prefill_does_not_materialize_a_buffer_per_token():
    """Repeated single-token appends without an eval between them must cost
    the two final buffers and nothing more."""

    heads, dim, prompt = 4, 256, 512
    keys = mx.random.normal((heads, prompt, dim)).astype(mx.float16)
    values = mx.random.normal((heads, prompt, dim)).astype(mx.float16)
    mx.eval(keys, values)

    cache = GrowingKVCache(
        max_context=prompt, num_kv_heads=heads, head_dim=dim, dtype=mx.float16
    )

    gc.collect()  # same reasoning as the test above: keep collection out of the window

    baseline = mx.get_active_memory()

    for i in range(prompt):
        cache.append(keys[:, i, :], values[:, i, :])

    mx.eval(cache._keys, cache._values)

    gc.collect()

    # Two buffers of 4*512*256*2 bytes = 1 MB each. Anything close to
    # prompt-many buffers would be hundreds of megabytes.
    grown = mx.get_active_memory() - baseline
    assert grown <= 8 * 2**20, (
        f"a {prompt}-token prefill added {grown / 2**20:.1f} MB; the cache itself is 2 MB"
    )


def test_allocation_follows_used_context():
    cache = GrowingKVCache(32768, NUM_KV_HEADS, HEAD_DIM)
    assert cache.allocated_size == 0

    cache.append(_token(1), _token(2))
    assert cache.allocated_size == 256
    assert cache.capacity == 32768


def test_batch_append_crosses_growth_boundaries_and_preserves_history():
    cache = GrowingKVCache(600, NUM_KV_HEADS, HEAD_DIM)
    keys = mx.broadcast_to(mx.arange(600)[None, :, None], (NUM_KV_HEADS, 600, HEAD_DIM))
    values = -keys

    for start, end in ((0, 255), (255, 258), (258, 599), (599, 600)):
        cache.append_many(keys[:, start:end], values[:, start:end])

        actual_keys, actual_values = cache.state()
        assert mx.array_equal(actual_keys, keys[:, :end]).item()
        assert mx.array_equal(actual_values, values[:, :end]).item()
        assert end <= cache.allocated_size <= min(end + 255, cache.capacity)


def test_failed_batch_append_leaves_history_unchanged():
    cache = GrowingKVCache(3, NUM_KV_HEADS, HEAD_DIM)
    cache.append(_token(1), _token(2))

    with pytest.raises(IndexError):
        cache.append_many(
            mx.zeros((NUM_KV_HEADS, 3, HEAD_DIM)), mx.zeros((NUM_KV_HEADS, 3, HEAD_DIM))
        )

    assert cache.size == 1

    keys, values = cache.state()
    assert mx.array_equal(keys[:, 0], _token(1)).item()
    assert mx.array_equal(values[:, 0], _token(2)).item()
