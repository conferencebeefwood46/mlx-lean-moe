"""On-demand tensor reads, and the per-layer expert cache above them.

Reads use ``os.pread`` rather than ``mmap``: these are cold reads, where
page-fault machinery measured ~29.6ms against ~8.2ms for four fresh experts.
"""

import os
from collections import OrderedDict
from collections.abc import Sequence
from concurrent.futures import Executor
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_lean_moe.config import QuantScheme
from mlx_lean_moe.model.moe_block import LinearWeights
from mlx_lean_moe.weights._shard_fds import ShardFdCache
from mlx_lean_moe.weights.safetensors_index import TensorLocation

# BF16 has no numpy dtype and is handled separately in `decode_tensor`.
_NUMPY_DTYPES = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
    "I64": np.int64,
    "I32": np.int32,
    "I16": np.int16,
    "I8": np.int8,
    "U64": np.uint64,
    "U32": np.uint32,
    "U16": np.uint16,
    "U8": np.uint8,
    "BOOL": np.bool_,
}

DEFAULT_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


# MLX's default stream is thread-local: an op built on a reader thread must
# name the stream that will evaluate it, or it fails at evaluation.
_DECODE_STREAM = mx.default_stream(mx.default_device())


def decode_on_this_thread() -> None:
    """Pin tensor decoding to the calling thread's stream, before building a
    model on it."""
    global _DECODE_STREAM
    _DECODE_STREAM = mx.default_stream(mx.default_device())


# Only quantization metadata. Norm weights, `A_log`, `dt_bias` and the
# convolution kernel are bfloat16 too, and each feeds float32 arithmetic.
_STORED_WIDTH_FIELDS = ("scales", "biases")


def keeps_stored_width(name: str) -> bool:
    return name.rpartition(".")[2] in _STORED_WIDTH_FIELDS


def decode_tensor(
    raw: bytes, dtype: str, shape: tuple[int, ...], *, stored_width: bool = False
) -> mx.array:
    if dtype == "BF16" and stored_width:
        # Reinterpreted rather than widened: MLX's quantized matmul reads
        # bf16 scales bit-identically, and widening doubles what they cost.
        packed = mx.array(np.frombuffer(raw, dtype=np.uint16).reshape(shape))
        return mx.view(packed, mx.bfloat16, stream=_DECODE_STREAM)

    if dtype == "BF16":
        # bfloat16 is the top 16 bits of a float32, so this widening is exact.
        as_u16 = np.frombuffer(raw, dtype=np.uint16)
        as_f32 = (as_u16.astype(np.uint32) << 16).view(np.float32)
        return mx.array(as_f32.reshape(shape))

    np_dtype = _NUMPY_DTYPES.get(dtype)
    if np_dtype is None:
        raise ValueError(f"unsupported safetensors dtype: {dtype!r}")
    array = np.frombuffer(raw, dtype=np_dtype).reshape(shape)
    return mx.array(array)


def _as_float32(array: mx.array | None) -> mx.array | None:
    if array is None or array.dtype == mx.float32:
        return array
    return array.astype(mx.float32)


def validate_linear_layout(
    index: dict[str, TensorLocation], name: str, quant: QuantScheme
) -> bool:
    """Validate a projection's packed shape; return whether it is quantized."""
    weight_name = f"{name}.weight"
    if weight_name not in index:
        raise KeyError(f"no weight tensor for {name}")
    scale_name = f"{name}.scales"
    bias_name = f"{name}.biases"
    has_scales = scale_name in index
    has_biases = bias_name in index
    if not has_scales:
        if has_biases:
            raise ValueError(f"{name} has biases but no scales")
        return False
    if quant.mode == "affine" and not has_biases:
        raise ValueError(f"affine-quantized {name} has no biases")

    weight = index[weight_name]
    scales = index[scale_name]
    if weight.shape[:-1] != scales.shape[:-1]:
        raise ValueError(
            f"{name} weight shape {weight.shape} is incompatible with scales shape {scales.shape}"
        )
    if has_biases and index[bias_name].shape != scales.shape:
        raise ValueError(
            f"{name} biases shape {index[bias_name].shape} does not match scales shape {scales.shape}"
        )
    logical_width = scales.shape[-1] * quant.group_size
    expected_packed_width = logical_width * quant.bits // 32
    if weight.shape[-1] != expected_packed_width:
        raise ValueError(
            f"{name} weight shape {weight.shape} does not match bits={quant.bits}, "
            f"group_size={quant.group_size}; expected packed width {expected_packed_width}"
        )
    return True


def read_linear(
    streamer: "TensorStreamer", name: str, quant: QuantScheme
) -> LinearWeights:
    """Read a quantized projection or a plain FP matrix from its real fields."""
    quantized = validate_linear_layout(streamer.index, name, quant)
    if not quantized:
        return LinearWeights(streamer.read_tensor(f"{name}.weight"), None)
    fields = {
        field
        for field in ("weight", "scales", "biases")
        if f"{name}.{field}" in streamer.index
    }
    tensors = {field: streamer.read_tensor(f"{name}.{field}") for field in fields}
    return LinearWeights(tensors, quant)


class TensorStreamer:
    """Reads whole tensors, or individual rows of one, out of a checkpoint's
    shards on demand. Everything a token needs besides its routed experts."""

    def __init__(self, model_dir: str | Path, index: dict[str, TensorLocation]) -> None:
        self.model_dir = Path(model_dir)
        self.index = index
        self._fds = ShardFdCache(self.model_dir)

    def read_tensor(self, name: str) -> mx.array:
        loc = self.index[name]
        fd = self._fds.fd_for_shard(loc.shard)
        raw = os.pread(fd, loc.length, loc.offset)
        return decode_tensor(
            raw, loc.dtype, loc.shape, stored_width=keeps_stored_width(name)
        )

    def read_rows(self, name: str, rows: Sequence[int]) -> mx.array:
        """Reads only the given rows, stacked in the order asked for, one
        read each: a prompt's tokens are scattered through the vocabulary."""
        if not rows:
            raise ValueError(f"no rows requested from {name}")
        loc = self.index[name]
        if len(loc.shape) < 2:
            raise ValueError(
                f"{name} has shape {loc.shape}, which has no rows to index"
            )

        num_rows = loc.shape[0]
        row_bytes, remainder = divmod(loc.length, num_rows)
        if remainder:
            raise ValueError(
                f"{name} is {loc.length} bytes over {num_rows} rows, which does not divide"
            )

        fd = self._fds.fd_for_shard(loc.shard)
        chunks = []
        for row in rows:
            if not 0 <= row < num_rows:
                raise IndexError(f"row {row} is outside {name}'s {num_rows} rows")
            chunks.append(os.pread(fd, row_bytes, loc.offset + row * row_bytes))
        return decode_tensor(
            b"".join(chunks),
            loc.dtype,
            (len(rows), *loc.shape[1:]),
            stored_width=keeps_stored_width(name),
        )

    def read_linear_rows(
        self,
        prefix: str,
        rows: Sequence[int],
        quant: QuantScheme,
        *,
        quantized: bool | None = None,
    ) -> mx.array:
        """Read and decode only selected rows of a quantized or dense table."""
        if quantized is None:
            quantized = validate_linear_layout(self.index, prefix, quant)
        if not quantized:
            return self.read_rows(f"{prefix}.weight", rows)
        fields = ["weight", "scales"]
        if f"{prefix}.biases" in self.index:
            fields.append("biases")
        tensors = {field: self.read_rows(f"{prefix}.{field}", rows) for field in fields}
        # `mx.dequantize` returns whatever dtype its scales carry, unlike the
        # quantized matmul, so bf16 here would round every row it produces.
        scales = _as_float32(tensors["scales"])
        biases = _as_float32(tensors.get("biases"))
        return mx.dequantize(
            tensors["weight"],
            scales=scales,
            biases=biases,
            group_size=quant.group_size,
            bits=quant.bits,
            mode=quant.mode,
        )

    def close(self) -> None:
        self._fds.close()


class CachedExpertLoader:
    """A fixed-size, least-recently-used cache of decoded experts, in front
    of a streamer. One instance per layer."""

    def __init__(
        self,
        streamer: TensorStreamer,
        max_size: int,
        executor: Executor | None = None,
    ) -> None:
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")
        self._streamer = streamer
        self._max_size = max_size
        self._executor = executor
        # Ordered by how recently each key was used, oldest first.
        self._cache: OrderedDict[tuple[int, int], dict] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def _touch(self, key: tuple[int, int]) -> None:
        self._cache.move_to_end(key)

    def _insert(self, key: tuple[int, int], value: dict) -> None:
        if len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)
        self._cache[key] = value

    def load_expert(self, layer: int, expert: int) -> dict[str, dict[str, mx.array]]:
        key = (layer, expert)
        cached = self._cache.get(key)
        if cached is not None:
            self._touch(key)
            self.hits += 1
            return cached

        self.misses += 1
        value = self._streamer.load_expert(layer, expert)
        self._insert(key, value)
        return value

    def load_many(
        self, layer: int, experts: list[int]
    ) -> list[dict[str, dict[str, mx.array]]]:
        """`load_expert` for each of `experts`, with cache-miss reads run
        concurrently on `executor` when one was given."""
        results: list[dict | None] = [None] * len(experts)
        misses: list[tuple[int, int]] = []  # (result position, expert id)

        for pos, expert in enumerate(experts):
            key = (layer, expert)
            cached = self._cache.get(key)
            if cached is not None:
                self._touch(key)
                self.hits += 1
                results[pos] = cached
            else:
                misses.append((pos, expert))

        if not misses:
            return results  # type: ignore[return-value]

        if self._executor is None or len(misses) == 1:
            for pos, expert in misses:
                self.misses += 1
                value = self._streamer.load_expert(layer, expert)
                self._insert((layer, expert), value)
                results[pos] = value
        else:
            miss_experts = [expert for _, expert in misses]
            loaded = self._streamer.load_experts_concurrently(
                layer, miss_experts, self._executor
            )
            for (pos, expert), value in zip(misses, loaded):
                self.misses += 1
                self._insert((layer, expert), value)
                results[pos] = value

        return results  # type: ignore[return-value]

    def __contains__(self, key: tuple[int, int]) -> bool:
        return key in self._cache

    def __len__(self) -> int:
        return len(self._cache)
