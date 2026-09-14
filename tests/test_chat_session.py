"""`ChatSession`'s core claim: feeding only each turn's new tokens through
the model must produce bit-identical tokens to reprocessing the whole
conversation every time.

One model is loaded for the module and `reset()` stands in for a fresh
session, which is what it clears; building per test cost 100s against 12.
"""

import json

import pytest
from conftest import QWEN3_5_MODEL_DIR

from mlx_lean_moe.config import model_config_from_hf
from mlx_lean_moe.runtime.generate import ChatSession, generate

pytestmark = pytest.mark.skipif(
    not (QWEN3_5_MODEL_DIR / "config.json").exists(),
    reason="the qwen3_5 validation checkpoint is not downloaded",
)

MAX_CONTEXT = 64


@pytest.fixture(scope="module")
def config():
    return model_config_from_hf(
        json.loads((QWEN3_5_MODEL_DIR / "config.json").read_text())
    )


@pytest.fixture(scope="module")
def _model(config):
    with ChatSession(QWEN3_5_MODEL_DIR, config, max_context=MAX_CONTEXT) as session:
        yield session


@pytest.fixture
def session(_model):
    """A session with no conversation behind it."""
    _model.reset()
    return _model


def test_incremental_turns_match_a_full_rebuild_bit_for_bit(session):
    """Turn 2 must be identical whether the cache was built across two
    `send()` calls or from the whole history in one fresh session."""
    turn1_prompt = [1, 2, 3]
    turn2_new = [4, 5]

    turn1_reply = list(session.send(turn1_prompt, max_new_tokens=2))
    full_turn2_prompt = turn1_prompt + turn1_reply + turn2_new
    incremental = list(session.send(full_turn2_prompt, max_new_tokens=2))

    session.reset()
    rebuilt = list(session.send(full_turn2_prompt, max_new_tokens=2))

    assert incremental == rebuilt


def test_second_send_only_prefills_the_new_suffix(session):
    turn1_prompt = [1, 2, 3]
    turn2_new = [4, 5]

    turn1_reply = list(session.send(turn1_prompt, max_new_tokens=2))
    assert session.last_prefill_len == len(turn1_prompt)

    list(session.send(turn1_prompt + turn1_reply + turn2_new, max_new_tokens=1))
    assert session.last_prefill_len == len(turn2_new)


def test_diverging_history_triggers_a_transparent_reset(session):
    """Tokens that don't extend what is committed must rebuild, and still
    produce what a fresh session would."""
    list(session.send([1, 2, 3], max_new_tokens=2))
    diverged_prompt = [9, 8, 7]  # shares no prefix with what's committed
    after_divergence = list(session.send(diverged_prompt, max_new_tokens=2))
    assert session.last_prefill_len == len(diverged_prompt)  # full rebuild

    session.reset()
    fresh = list(session.send(diverged_prompt, max_new_tokens=2))

    assert after_divergence == fresh


def test_stop_token_is_never_committed_to_history(session):
    """Breaking out early, as a caller stopping on an eos id does, must skip
    committing that token."""
    for _ in session.send([1, 2, 3], max_new_tokens=5):
        break  # stop after the very first generated token, like an eos hit
    assert session._committed == [1, 2, 3]


def test_generate_one_shot_is_unaffected_by_chat_session_refactor(config):
    """generate() sits on ChatSession; its own one-shot behaviour must not
    change. The one test here that pays for a load."""
    tokens = list(generate(QWEN3_5_MODEL_DIR, config, [1, 2, 3], max_new_tokens=2))
    assert len(tokens) == 2
    assert all(isinstance(t, int) for t in tokens)


# Which reset path is taken, not what it computes, so these run against a
# fake model patched over `_build_model`.


class _FakeModelWithResetCache:
    def __init__(self):
        self.reset_cache_calls = 0
        self.closed = False

    def reset_cache(self):
        self.reset_cache_calls += 1

    def close(self):
        self.closed = True


class _FakeModelWithoutResetCache:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def test_chat_session_reset_prefers_cheap_reset_cache_when_available(monkeypatch):
    from mlx_lean_moe.runtime import generate as generate_module

    fake_model = _FakeModelWithResetCache()
    monkeypatch.setattr(generate_module, "_build_model", lambda *a, **k: fake_model)

    session = ChatSession("fake/dir", object(), max_context=16)
    session.reset()

    assert fake_model.reset_cache_calls == 1
    assert not fake_model.closed  # the cheap path never closes/rebuilds the model
    assert session.model is fake_model


def test_chat_session_reset_falls_back_to_a_full_rebuild_without_reset_cache(
    monkeypatch,
):
    from mlx_lean_moe.runtime import generate as generate_module

    built: list[_FakeModelWithoutResetCache] = []

    def fake_build_model(*_args, **_kwargs):
        model = _FakeModelWithoutResetCache()
        built.append(model)
        return model

    monkeypatch.setattr(generate_module, "_build_model", fake_build_model)

    session = ChatSession("fake/dir", object(), max_context=16)
    first_model = session.model
    session.reset()

    assert first_model.closed
    assert session.model is not first_model
    assert len(built) == 2
