"""Token-by-token generation, one-shot or as a persistent multi-turn
:class:`ChatSession`.

The explicit `mx.clear_cache()` calls bound the Metal allocator's reuse
cache.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import mlx.core as mx

from mlx_lean_moe.config import GenerativeModel, model_class_for
from mlx_lean_moe.runtime.sampling import Sampler, Sampling


def _build_model(
    model_dir: str | Path,
    config: Any,
    max_context: int,
    expert_cache_size_per_layer: int | None,
) -> GenerativeModel:
    """The only place an architecture-specific model class is instantiated,
    and it names none of them."""
    model_cls = model_class_for(config)
    return model_cls(
        model_dir,
        config,
        max_context,
        expert_cache_size_per_layer=expert_cache_size_per_layer,
    )


class ChatSession:
    """Keeps one model and its caches alive across turns; :meth:`send` takes
    a turn's full context but runs only the new suffix."""

    def __init__(
        self,
        model_dir: str | Path,
        config: Any,
        max_context: int,
        expert_cache_size_per_layer: int | None = None,
    ) -> None:
        self.model_dir = model_dir
        self.config = config
        self.max_context = max_context
        self.expert_cache_size_per_layer = expert_cache_size_per_layer
        self.model = _build_model(model_dir, config, max_context, expert_cache_size_per_layer)
        self._committed: list[int] = []
        self._logits: mx.array | None = None
        self.last_prefill_len = 0

    def _common_prefix_len(self, token_ids: list[int]) -> int:
        limit = min(len(self._committed), len(token_ids))
        n = 0
        while n < limit and self._committed[n] == token_ids[n]:
            n += 1
        return n

    def reset(self) -> None:
        """Drops conversation state and starts a fresh cache.

        A KV cache only moves forward, so a diverging history starts over.
        """
        reset_cache = getattr(self.model, "reset_cache", None)
        if callable(reset_cache):
            reset_cache()
        else:
            self.model.close()
            self.model = _build_model(
                self.model_dir,
                self.config,
                self.max_context,
                self.expert_cache_size_per_layer,
            )
        self._committed = []
        self._logits = None
        self.last_prefill_len = 0

    def send(
        self,
        token_ids: list[int],
        max_new_tokens: int,
        clear_cache_every: int = 8,
        sampling: Sampling | None = None,
    ) -> Iterator[int]:
        """Yields token ids one at a time; `token_ids` is this turn's full
        context. A token the caller stops on is never committed."""
        common = self._common_prefix_len(token_ids)
        if common < len(self._committed):
            self.reset()
            common = 0
        new_suffix = token_ids[common:]
        self.last_prefill_len = len(new_suffix)
        if new_suffix:
            self._logits = self.model.prefill(new_suffix)
            self._committed = list(token_ids)
        elif self._logits is None:
            raise ValueError("ChatSession.send() got no new tokens and there's no prior turn to continue")

        pick = Sampler(sampling)

        for step in range(max_new_tokens):
            next_token = pick(self._logits)
            yield next_token
            if (step + 1) % clear_cache_every == 0:
                mx.clear_cache()
            self._logits = self.model(next_token)
            self._committed.append(next_token)

    def close(self) -> None:
        self.model.close()

    def __enter__(self) -> "ChatSession":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def generate(
    model_dir: str | Path,
    config: Any,
    prompt_tokens: list[int],
    max_new_tokens: int,
    max_context: int | None = None,
    clear_cache_every: int = 8,
    expert_cache_size_per_layer: int | None = None,
    sampling: Sampling | None = None,
) -> Iterator[int]:
    """One turn through a fresh :class:`ChatSession`, closed afterwards; use
    `ChatSession` itself to reuse the cache across turns."""
    max_context = max_context or (len(prompt_tokens) + max_new_tokens)
    with ChatSession(model_dir, config, max_context, expert_cache_size_per_layer) as session:
        yield from session.send(prompt_tokens, max_new_tokens, clear_cache_every, sampling)
