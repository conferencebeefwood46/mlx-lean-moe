"""Cross-checks the gated-delta recurrence against `mlx_lm`'s own
implementation, a dev-only dependency never imported at runtime. Synthetic
inputs: the recurrence is pure math over arbitrary arrays.
"""

import mlx.core as mx
import numpy as np
import pytest

from mlx_lean_moe.model.qwen3_5.gated_delta import (
    forget_gate,
    gated_delta_scan,
    gated_delta_step,
)

mlx_lm_gated_delta = pytest.importorskip("mlx_lm.models.gated_delta")

HK, HV, DK, DV = (
    2,
    6,
    16,
    8,
)  # small, but with num_value_heads a multiple of num_key_heads like the real model


def _inputs(seed: int, seq_len: int):
    rng = np.random.default_rng(seed)
    f32 = lambda *shape: mx.array(rng.standard_normal(shape).astype(np.float32))

    return {
        "q": f32(seq_len, HK, DK),
        "k": f32(seq_len, HK, DK),
        "v": f32(seq_len, HV, DV),
        "a": f32(seq_len, HV),
        "b": f32(seq_len, HV),
        "A_log": f32(HV),
        "dt_bias": f32(HV),
    }


@pytest.mark.parametrize("seq_len", [1, 5])
def test_scan_matches_mlx_lm_reference(seq_len):
    """Against mlx_lm's `gated_delta_update` ops path, its Metal kernel
    being an optimization of the same math."""

    x = _inputs(seed=0, seq_len=seq_len)
    state = mx.zeros((HV, DV, DK), dtype=mx.float32)

    ours_y, ours_state = gated_delta_scan(
        x["q"],
        x["k"],
        x["v"],
        forget_gate(x["a"], x["A_log"], x["dt_bias"]),
        mx.sigmoid(x["b"]),
        state,
    )

    # mlx_lm works in batch-first shapes; ours is single-sequence.
    ref_y, ref_state = mlx_lm_gated_delta.gated_delta_update(
        x["q"][None],
        x["k"][None],
        x["v"][None],
        x["a"][None],
        x["b"][None],
        x["A_log"],
        x["dt_bias"],
        state=state[None],
        use_kernel=False,
    )

    mx.eval(ours_y, ours_state, ref_y, ref_state)

    assert mx.allclose(ours_y, ref_y[0], rtol=1e-4, atol=1e-5).item()
    assert mx.allclose(ours_state, ref_state[0], rtol=1e-4, atol=1e-5).item()


def test_scan_continues_from_a_carried_state():
    """5 positions in one call must equal 2 then 3 carrying the state, which
    is what decode does after prefill."""

    x = _inputs(seed=1, seq_len=5)
    g, beta = forget_gate(x["a"], x["A_log"], x["dt_bias"]), mx.sigmoid(x["b"])
    zero = mx.zeros((HV, DV, DK), dtype=mx.float32)

    whole_y, whole_state = gated_delta_scan(x["q"], x["k"], x["v"], g, beta, zero)

    first_y, mid_state = gated_delta_scan(
        x["q"][:2], x["k"][:2], x["v"][:2], g[:2], beta[:2], zero
    )
    second_y, final_state = gated_delta_scan(
        x["q"][2:], x["k"][2:], x["v"][2:], g[2:], beta[2:], mid_state
    )
    split_y = mx.concatenate([first_y, second_y])

    mx.eval(whole_y, whole_state, split_y, final_state)

    assert mx.allclose(whole_y, split_y, rtol=1e-5, atol=1e-6).item()
    assert mx.allclose(whole_state, final_state, rtol=1e-5, atol=1e-6).item()


def test_single_step_matches_the_scan_of_one():
    x = _inputs(seed=2, seq_len=1)
    g, beta = forget_gate(x["a"], x["A_log"], x["dt_bias"]), mx.sigmoid(x["b"])
    state = mx.zeros((HV, DV, DK), dtype=mx.float32)

    scan_y, scan_state = gated_delta_scan(x["q"], x["k"], x["v"], g, beta, state)
    # gated_delta_step takes q/k already repeated out to the value heads.
    repeat = HV // HK
    step_y, step_state = gated_delta_step(
        mx.repeat(x["q"][0], repeat, axis=0),
        mx.repeat(x["k"][0], repeat, axis=0),
        x["v"][0],
        g[0],
        beta[0],
        state,
    )

    mx.eval(scan_y, scan_state, step_y, step_state)

    assert mx.allclose(scan_y[0], step_y, rtol=1e-6, atol=1e-7).item()
    assert mx.allclose(scan_state, step_state, rtol=1e-6, atol=1e-7).item()


def test_forget_gate_is_a_decay_in_the_unit_interval():
    """g multiplies the state every step, so it has to stay in (0, 1] --
    anything above 1 would make the recurrence blow up over a long context."""

    x = _inputs(seed=3, seq_len=4)
    g = forget_gate(x["a"], x["A_log"], x["dt_bias"])
    mx.eval(g)

    assert bool((g > 0).all().item())
    assert bool((g <= 1.0).all().item())
