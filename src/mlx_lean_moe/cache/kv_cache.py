"""The per-layer key/value cache for softmax-attention layers.

Head-major, ``(num_kv_heads, allocated_size, head_dim)``, which is what
``scaled_dot_product_attention`` takes.
"""

from typing import Protocol

import mlx.core as mx


class KVCache(Protocol):
    capacity: int

    def append(self, key: mx.array, value: mx.array) -> None: ...

    def append_many(self, keys: mx.array, values: mx.array) -> None: ...

    @property
    def size(self) -> int: ...

    def state(self) -> tuple[mx.array, mx.array]:
        """Keys/values for every token currently held, in chronological order."""

        ...


class GrowingKVCache:
    """A linear buffer growing in 256-token blocks up to ``max_context``,
    which is the hard limit: appending past it raises rather than wraps."""

    def __init__(
        self,
        max_context: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: mx.Dtype = mx.float16,
    ) -> None:
        if max_context <= 0:
            raise ValueError(f"max_context must be positive, got {max_context}")

        self.capacity = max_context
        self._keys = mx.zeros((num_kv_heads, 0, head_dim), dtype=dtype)
        self._values = mx.zeros((num_kv_heads, 0, head_dim), dtype=dtype)
        self._written = 0

    def append(self, key: mx.array, value: mx.array) -> None:
        self.append_many(key[:, None, :], value[:, None, :])

    def append_many(self, keys: mx.array, values: mx.array) -> None:
        """Append a head-major ``(heads, tokens, dim)`` batch in two writes,
        checked before either buffer is touched."""

        if keys.ndim != 3 or values.shape != keys.shape:
            raise ValueError(
                "keys and values must have matching (heads, tokens, dim) shapes"
            )

        if keys.shape[0] != self._keys.shape[0] or keys.shape[2] != self._keys.shape[2]:
            raise ValueError(
                "keys and values do not match the cache head count or dimension"
            )

        end = self._written + keys.shape[1]

        if end > self.capacity:
            raise IndexError(
                f"GrowingKVCache exceeded its capacity of {self.capacity} tokens"
            )

        if end == self._written:
            return

        if end > self.allocated_size:
            allocated = min(self.capacity, ((end + 255) // 256) * 256)
            shape = (
                self._keys.shape[0],
                allocated - self.allocated_size,
                self._keys.shape[2],
            )
            self._keys = mx.concatenate(
                [self._keys, mx.zeros(shape, dtype=self._keys.dtype)], axis=1
            )
            self._values = mx.concatenate(
                [self._values, mx.zeros(shape, dtype=self._values.dtype)], axis=1
            )

        self._keys[:, self._written : end, :] = keys
        self._values[:, self._written : end, :] = values
        self._written = end

    @property
    def allocated_size(self) -> int:
        return self._keys.shape[1]

    @property
    def size(self) -> int:
        return self._written

    def state(self) -> tuple[mx.array, mx.array]:
        n = self._written
        return self._keys[:, :n, :], self._values[:, :n, :]
