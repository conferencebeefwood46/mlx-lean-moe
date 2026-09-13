"""Token selection.

Sampling has no reference to compare against -- a wrong nucleus cut still
produces plausible text -- so these check distributions and invariants.
"""

import math
from collections import Counter

import mlx.core as mx
import pytest

from mlx_lean_moe.runtime.sampling import Sampler, Sampling, _nucleus


def _draws(sampler, logits, n=400):
    return Counter(sampler(logits) for _ in range(n))


def test_the_default_is_the_greedy_behaviour_that_came_before():
    """Every numerical check in this project rests on the same prompt giving
    the same tokens, so the default has to stay exactly argmax."""
    sampler = Sampler()
    assert sampler.sampling.greedy
    logits = mx.array([0.1, 5.0, 0.2, 4.9])
    assert [sampler(logits) for _ in range(5)] == [1] * 5


def test_zero_temperature_is_argmax_not_an_approximation_of_it():
    sampler = Sampler(Sampling(temperature=0.0, top_p=0.5, seed=3))
    logits = mx.array([1.0, 3.0, 2.0])
    assert [sampler(logits) for _ in range(5)] == [1] * 5


def test_a_low_temperature_concentrates_and_a_high_one_spreads():
    logits = mx.array([0.0, 1.0, 2.0])
    cold = _draws(Sampler(Sampling(temperature=0.1, seed=1)), logits)
    hot = _draws(Sampler(Sampling(temperature=5.0, seed=1)), logits)
    assert cold[2] / cold.total() > 0.95
    # Hot enough that the least likely token is still drawn regularly.
    assert hot[0] / hot.total() > 0.15
    assert len(hot) == 3


def test_sampling_can_choose_something_other_than_the_argmax():
    logits = mx.array([2.0, 2.1, 2.0])
    assert len(_draws(Sampler(Sampling(temperature=1.0, seed=11)), logits)) == 3


def test_the_same_seed_replays_and_a_different_one_does_not():
    logits = mx.array([1.0, 1.0, 1.0, 1.0, 1.0])
    first = [Sampler(Sampling(temperature=1.0, seed=42))(logits) for _ in range(1)]
    again = [Sampler(Sampling(temperature=1.0, seed=42))(logits) for _ in range(1)]
    assert first == again

    run_a = [x for x in _draws(Sampler(Sampling(temperature=1.0, seed=42)), logits, 30).elements()]
    run_b = [x for x in _draws(Sampler(Sampling(temperature=1.0, seed=42)), logits, 30).elements()]
    assert run_a == run_b


def test_a_seeded_sampler_does_not_repeat_one_token_forever():
    """A key used twice draws the same token twice."""
    logits = mx.array([1.0] * 8)
    drawn = _draws(Sampler(Sampling(temperature=1.0, seed=5)), logits, 60)
    assert len(drawn) > 1


def test_unseeded_sampling_still_varies():
    logits = mx.array([1.0] * 8)
    assert len(_draws(Sampler(Sampling(temperature=1.0)), logits, 60)) > 1


def test_top_p_never_chooses_what_it_cut_away():
    # Probabilities are roughly 0.87, 0.12, 0.016 -- 0.9 reaches into the
    # second, and must never reach the third.
    logits = mx.array([4.0, 2.0, 0.0])
    drawn = _draws(Sampler(Sampling(temperature=1.0, top_p=0.9, seed=2)), logits, 500)
    assert 2 not in drawn
    assert set(drawn) == {0, 1}


def test_top_p_keeps_the_token_that_crosses_the_threshold():
    """Cutting the crossing token instead would leave a distribution whose
    likeliest token alone exceeds `top_p` with nothing to sample."""
    logits = mx.array([10.0, 0.0, 0.0])  # the first is already over 0.99
    drawn = _draws(Sampler(Sampling(temperature=1.0, top_p=0.5, seed=4)), logits, 50)
    assert set(drawn) == {0}


def test_top_p_of_one_keeps_everything():
    logits = mx.array([1.0, 1.0, 1.0])
    kept = _nucleus(logits, 1.0)
    assert not bool(mx.any(mx.isinf(kept)).item())


def test_the_nucleus_leaves_surviving_logits_untouched():
    logits = mx.array([4.0, 2.0, 0.0])
    kept = _nucleus(logits, 0.9)
    assert math.isclose(float(kept[0].item()), 4.0, abs_tol=1e-5)
    assert math.isclose(float(kept[1].item()), 2.0, abs_tol=1e-5)
    assert kept[2].item() == -math.inf


def test_the_nucleus_cuts_by_probability_not_by_position():
    """Logits arrive in vocabulary order, not sorted order, so the cut has to
    find the likely ones wherever they are."""
    logits = mx.array([0.0, 4.0, 0.0, 2.0])  # the nucleus is indices 1 and 3
    kept = _nucleus(logits, 0.9)
    assert float(kept[1].item()) == 4.0
    assert float(kept[3].item()) == 2.0
    assert kept[0].item() == -math.inf
    assert kept[2].item() == -math.inf


@pytest.mark.parametrize(
    ("temperature", "top_p"),
    [(-0.1, 1.0), (1.0, 0.0), (1.0, -0.5), (1.0, 1.5)],
)
def test_impossible_settings_are_refused_at_the_edge(temperature, top_p):
    with pytest.raises(ValueError):
        Sampling(temperature=temperature, top_p=top_p)


def test_logit_bias_can_make_an_unlikely_token_certain():
    sampler = Sampler(Sampling(logit_bias={2: 100.0}))
    assert sampler(mx.array([5.0, 4.0, -5.0])) == 2


def test_logit_bias_can_ban_the_token_that_would_have_won():
    sampler = Sampler(Sampling(logit_bias={0: -100.0}))
    assert sampler(mx.array([5.0, 4.0, 3.0])) == 1


def test_logit_bias_applies_under_greedy_decoding():
    """Greedy is not "no sampling settings": a bias moves which token is the
    argmax, so skipping it there would honour the parameter only sometimes."""
    sampler = Sampler(Sampling(temperature=0.0, logit_bias={1: 50.0}))
    assert sampler.sampling.greedy
    assert sampler(mx.array([5.0, 4.0, 3.0])) == 1


def test_frequency_penalty_grows_until_it_balances_the_two():
    logits = mx.array([3.0, 2.0])
    sampler = Sampler(Sampling(frequency_penalty=0.75))
    assert [sampler(logits) for _ in range(6)] == [0, 0, 1, 0, 1, 0]

    drawn = _draws(Sampler(Sampling(frequency_penalty=0.75)), logits, 40)
    assert drawn[0] > 15 and drawn[1] > 15


def test_presence_penalty_is_paid_once_and_never_again():
    """The same logits as the frequency penalty above, for the contrast."""
    logits = mx.array([3.0, 2.0])
    sampler = Sampler(Sampling(presence_penalty=1.5))
    assert [sampler(logits) for _ in range(6)] == [0, 1, 0, 0, 0, 0]

    drawn = _draws(Sampler(Sampling(presence_penalty=1.5)), logits, 40)
    assert drawn[0] == 39 and drawn[1] == 1


def test_penalties_count_only_what_was_generated():
    """The prompt is the user's own text; penalising it would push the model
    away from the subject it was asked about."""
    sampler = Sampler(Sampling(frequency_penalty=2.0))
    assert sampler(mx.array([3.0, 2.0])) == 0


def test_no_penalty_leaves_the_greedy_path_untouched():
    logits = mx.array([3.0, 2.0])
    sampler = Sampler(Sampling())
    assert not sampler.sampling.shifts_logits
    assert [sampler(logits) for _ in range(4)] == [0, 0, 0, 0]


def test_a_penalty_and_a_bias_on_one_token_both_apply():
    """On the same logit: the bias alone never flips this pair, the penalty
    alone flips it on the fourth step, together on the third."""
    logits = mx.array([3.0, 2.0])
    assert [Sampler(Sampling(logit_bias={0: -0.4}))(logits) for _ in range(4)] == [0, 0, 0, 0]

    penalty_only = Sampler(Sampling(frequency_penalty=0.4))
    assert [penalty_only(logits) for _ in range(4)] == [0, 0, 0, 1]

    both = Sampler(Sampling(frequency_penalty=0.4, logit_bias={0: -0.4}))
    assert [both(logits) for _ in range(4)] == [0, 0, 1, 0]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("frequency_penalty", 2.5),
        ("frequency_penalty", -2.5),
        ("presence_penalty", 3.0),
        ("presence_penalty", -3.0),
    ],
)
def test_penalties_outside_the_documented_range_are_refused(field, value):
    with pytest.raises(ValueError, match=field):
        Sampling(**{field: value})


@pytest.mark.parametrize("bias", [100.5, -100.5])
def test_a_bias_outside_the_documented_range_is_refused(bias):
    with pytest.raises(ValueError, match="logit_bias"):
        Sampling(logit_bias={7: bias})


def test_each_choice_of_a_seeded_request_moves_the_seed():
    """`n` completions from one seed would otherwise be the same completion
    `n` times, while the request as a whole stays reproducible."""
    base = Sampling(temperature=1.0, seed=11)
    assert base.for_choice(0) is base
    assert base.for_choice(1).seed == 12
    assert base.for_choice(3).seed == 14

    logits = mx.array([1.0] * 16)
    draws = {tuple(_draws(Sampler(base.for_choice(i)), logits, 8).elements()) for i in range(4)}
    assert len(draws) == 4


def test_an_unseeded_request_leaves_every_choice_unseeded():
    base = Sampling(temperature=1.0)
    assert base.for_choice(2).seed is None
