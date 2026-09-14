"""Choosing the next token from a row of logits.

Greedy is the default and `temperature=0` is exactly `argmax`, which is what
the project's numerical checks rest on.
"""

from collections import Counter
from dataclasses import dataclass, field, replace

import mlx.core as mx

PENALTY_LIMIT = 2.0
BIAS_LIMIT = 100.0


@dataclass(frozen=True, slots=True)
class Sampling:
    """How to pick a token; the default is greedy."""

    temperature: float = 0.0
    top_p: float = 1.0
    seed: int | None = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    logit_bias: dict[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError(
                f"temperature must not be negative, got {self.temperature}"
            )
        if not 0 < self.top_p <= 1:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        for name in ("frequency_penalty", "presence_penalty"):
            value = getattr(self, name)
            if not -PENALTY_LIMIT <= value <= PENALTY_LIMIT:
                raise ValueError(
                    f"{name} must be in [-{PENALTY_LIMIT}, {PENALTY_LIMIT}], got {value}"
                )
        for token, bias in self.logit_bias.items():
            if not -BIAS_LIMIT <= bias <= BIAS_LIMIT:
                raise ValueError(
                    f"logit_bias[{token}] must be in [-{BIAS_LIMIT}, {BIAS_LIMIT}], got {bias}"
                )

    @property
    def greedy(self) -> bool:
        return self.temperature == 0

    @property
    def shifts_logits(self) -> bool:
        """Whether anything here moves a logit before it is read. Greedy
        decoding honours these too: they change which token is the argmax."""
        return bool(self.frequency_penalty or self.presence_penalty or self.logit_bias)

    def for_choice(self, index: int) -> "Sampling":
        """The settings for one of a request's `n` completions. Each moves the
        seed, which reused would return the same completion `n` times."""
        if index == 0 or self.seed is None:
            return self
        return replace(self, seed=self.seed + index)


class Sampler:
    """Picks tokens, splitting the PRNG key on each step: a key used twice
    draws the same token twice."""

    def __init__(self, sampling: Sampling | None = None) -> None:
        self.sampling = sampling or Sampling()
        self._key = (
            None if self.sampling.seed is None else mx.random.key(self.sampling.seed)
        )
        self._produced: Counter[int] = Counter()

    def __call__(self, logits: mx.array) -> int:
        row = self._shifted(logits) if self.sampling.shifts_logits else logits

        if self.sampling.greedy:
            token = int(mx.argmax(row).item())
        else:
            scaled = row.reshape(-1).astype(mx.float32) / self.sampling.temperature
            if self.sampling.top_p < 1:
                scaled = _nucleus(scaled, self.sampling.top_p)

            key = None
            if self._key is not None:
                self._key, key = mx.random.split(self._key)
            token = int(mx.random.categorical(scaled, key=key).item())

        self._produced[token] += 1
        return token

    def _shifted(self, logits: mx.array) -> mx.array:
        """Logits with `logit_bias` added and the penalties subtracted. The
        penalties count only what this sampler produced, not the prompt."""
        sampling = self.sampling
        deltas: dict[int, float] = dict(sampling.logit_bias)
        if sampling.frequency_penalty or sampling.presence_penalty:
            for token, count in self._produced.items():
                penalty = sampling.frequency_penalty * count + sampling.presence_penalty
                deltas[token] = deltas.get(token, 0.0) - penalty

        if not deltas:
            return logits
        row = logits.reshape(-1).astype(mx.float32)
        ids = mx.array(list(deltas))
        row[ids] = row[ids] + mx.array(list(deltas.values()), dtype=mx.float32)
        return row


def _nucleus(logits: mx.array, top_p: float) -> mx.array:
    """The smallest set of tokens whose cumulative probability reaches
    `top_p`, the crossing token included, or a peaked one would keep none."""
    order = mx.argsort(-logits)
    ordered = logits[order]
    kept = mx.cumsum(mx.softmax(ordered)) - mx.softmax(ordered) < top_p

    masked = mx.where(kept, ordered, -mx.inf)
    restored = mx.zeros_like(masked)
    restored[order] = masked
    return restored
