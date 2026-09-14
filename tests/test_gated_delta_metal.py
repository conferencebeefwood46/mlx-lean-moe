"""Validate the shipped Metal scan against an independent float64 recurrence."""

import mlx.core as mx
import numpy as np
import pytest

import mlx_lean_moe.model.qwen3_5.gated_delta as delta


def _inputs(length, key_dim=128, value_dim=17, key_heads=2, value_heads=4):
    rng = np.random.default_rng(42)
    q = rng.normal(size=(length, key_heads, key_dim))
    k = rng.normal(size=q.shape)
    q /= np.linalg.norm(q, axis=-1, keepdims=True) * key_dim**0.5
    k /= np.linalg.norm(k, axis=-1, keepdims=True)
    v = rng.normal(size=(length, value_heads, value_dim))
    # Slow/absent decay exercises carried-state error over long contexts.
    g = rng.uniform(0.98, 1, size=(length, value_heads))
    g[:, 0] = 1
    beta = rng.uniform(0, 1, size=g.shape)
    state = rng.normal(size=(value_heads, value_dim, key_dim)) * 0.1
    return [mx.array(a.astype(np.float32)) for a in (q, k, v, g, beta, state)]


def _reference(inputs):
    q, k, v, g, beta, state = [np.array(a).astype(np.float64) for a in inputs]
    repeat = v.shape[1] // q.shape[1]
    q, k = np.repeat(q, repeat, axis=1), np.repeat(k, repeat, axis=1)
    output = []
    for t in range(len(q)):
        state *= g[t, :, None, None]
        read = np.einsum("hvk,hk->hv", state, k[t])
        correction = (v[t] - read) * beta[t, :, None]
        state += np.einsum("hv,hk->hvk", correction, k[t])
        output.append(np.einsum("hvk,hk->hv", state, q[t]))
    return np.stack(output), state


@pytest.mark.parametrize(
    "shape",
    [(1, 32, 5, 3, 6), (7, 64, 32, 2, 2), (17, 128, 128, 16, 32), (129, 256, 7, 2, 6)],
)
@pytest.mark.parametrize("strided", [False, True])
def test_kernel_matches_float64_reference_and_preserves_input_state(
    shape, strided, monkeypatch
):
    inputs = _inputs(*shape)
    if strided:
        inputs = [mx.stack([a, a], axis=-1)[..., 0] for a in inputs]
    expected_y, expected_state = _reference(inputs)
    initial_state = np.array(inputs[-1]).copy()

    def unexpected_fallback(*args):
        pytest.fail("supported float32 GPU inputs must use the Metal kernel")

    monkeypatch.setattr(delta, "_gated_delta_scan_ops", unexpected_fallback)
    y, state = delta.gated_delta_scan(*inputs)
    mx.eval(y, state)
    assert y.dtype == state.dtype == mx.float32
    np.testing.assert_allclose(np.array(y), expected_y, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(np.array(state), expected_state, rtol=2e-5, atol=2e-6)
    np.testing.assert_array_equal(np.array(inputs[-1]), initial_state)


def test_long_scan_and_prefill_then_decode_preserve_carried_state():
    inputs = _inputs(2048)
    expected_y, expected_state = _reference(inputs)
    whole_y, whole_state = delta.gated_delta_scan(*inputs)
    state = inputs[-1]
    outputs = []
    # Uneven prefill chunks, then 128 single-token decode steps.
    bounds = [0, 1, 128, 383, 1024, 1920, *range(1921, 2049)]
    for start, end in zip(bounds, bounds[1:]):
        y, state = delta.gated_delta_scan(*(a[start:end] for a in inputs[:-1]), state)
        mx.eval(y, state)
        outputs.append(y)
    split_y = mx.concatenate(outputs)
    mx.eval(whole_y, whole_state, split_y)
    for actual in (whole_y, split_y):
        np.testing.assert_allclose(np.array(actual), expected_y, rtol=2e-5, atol=2e-6)
    for actual in (whole_state, state):
        np.testing.assert_allclose(
            np.array(actual), expected_state, rtol=2e-5, atol=2e-6
        )


@pytest.mark.parametrize(
    "case",
    ["float16", "bfloat16", "mixed", "small_head", "cpu", "no_metal", "disabled"],
)
def test_fallback_preserves_previous_dtype_and_results(case, monkeypatch):
    inputs = _inputs(5, key_dim=16 if case == "small_head" else 128)
    if case in ("float16", "bfloat16"):
        inputs = [a.astype(getattr(mx, case)) for a in inputs]
    elif case == "mixed":
        inputs[2] = inputs[2].astype(mx.float16)

    def unexpected_kernel(*args):
        pytest.fail(f"{case} must use the ops fallback")

    monkeypatch.setattr(delta, "gated_delta_scan_metal", unexpected_kernel)
    if case == "no_metal":
        monkeypatch.setattr(mx.metal, "is_available", lambda: False)
    device = mx.default_device()
    try:
        if case == "cpu":
            mx.set_default_device(mx.cpu)
        actual = delta.gated_delta_scan(*inputs, use_kernel=case != "disabled")
        expected = delta.gated_delta_scan(*inputs, use_kernel=False)
        mx.eval(actual, expected)
        for got, want in zip(actual, expected):
            assert got.dtype == want.dtype
            assert mx.array_equal(got, want).item()
    finally:
        mx.set_default_device(device)


@pytest.mark.parametrize("invalid", ["k", "v", "g", "beta", "state", "heads", "empty"])
def test_invalid_shapes_are_rejected_before_kernel_dispatch(invalid, monkeypatch):
    inputs = _inputs(5)
    index = {"k": 1, "v": 2, "g": 3, "beta": 4, "state": 5}
    if invalid in index:
        inputs[index[invalid]] = inputs[index[invalid]][:1]
    elif invalid == "heads":
        inputs[0] = mx.repeat(inputs[0][:, :1], 3, axis=1)
        inputs[1] = inputs[0]
    else:
        inputs = [a[:0] for a in inputs[:-1]] + inputs[-1:]

    def unexpected_kernel(*args):
        pytest.fail("invalid shapes reached Metal")

    monkeypatch.setattr(delta, "gated_delta_scan_metal", unexpected_kernel)
    with pytest.raises(ValueError):
        delta.gated_delta_scan(*inputs)
