"""A byte-offset index over a checkpoint's safetensors shards.

Maps a tensor name to the shard, dtype, shape and byte range holding it, so
a reader can fetch one tensor without opening the rest.
"""

import json
import struct
from dataclasses import dataclass
from pathlib import Path

_INDEX_FILENAME = "model.safetensors.index.json"
_SIDECAR_FILENAME = ".mlx_lean_moe_index.json"
_SIDECAR_VERSION = 1


@dataclass(frozen=True, slots=True)
class TensorLocation:
    shard: str  # filename, relative to the checkpoint directory
    dtype: str  # safetensors dtype string, e.g. "F16", "U32", "BF16"
    shape: tuple[int, ...]
    offset: int  # absolute byte offset within `shard`
    length: int  # byte length of this tensor's raw data


def _parse_shard_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
    data_start = 8 + header_len
    return header, data_start


def _index_shard(shard_path: Path, shard_name: str) -> dict[str, TensorLocation]:
    header, data_start = _parse_shard_header(shard_path)
    index: dict[str, TensorLocation] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        index[name] = TensorLocation(
            shard=shard_name,
            dtype=meta["dtype"],
            shape=tuple(meta["shape"]),
            offset=data_start + start,
            length=end - start,
        )
    return index


def _discover_shards(model_dir: Path) -> list[str]:
    index_json = model_dir / _INDEX_FILENAME
    if index_json.exists():
        weight_map = json.loads(index_json.read_text())["weight_map"]
        return sorted(set(weight_map.values()))
    single = model_dir / "model.safetensors"
    if single.exists():
        return [single.name]
    raise FileNotFoundError(f"no {_INDEX_FILENAME} or model.safetensors found under {model_dir}")


def checkpoint_is_complete(model_dir: str | Path) -> bool:
    """Whether every shard is present and as long as its own header says,
    decided offline. Not a hash check; `download.py` verifies sha256."""
    model_dir = Path(model_dir)
    try:
        shards = _discover_shards(model_dir)
    except (FileNotFoundError, KeyError, ValueError):
        return False

    for shard_name in shards:
        path = model_dir / shard_name
        if not path.exists():
            return False
        try:
            header, data_start = _parse_shard_header(path)
        except (OSError, struct.error, json.JSONDecodeError):
            # Too short to even hold a header, or the header itself landed
            # truncated mid-JSON.
            return False
        data_end = max(
            (meta["data_offsets"][1] for name, meta in header.items() if name != "__metadata__"),
            default=0,
        )
        if path.stat().st_size < data_start + data_end:
            return False
    return True


def build_index(model_dir: str | Path, *, use_cache: bool = True) -> dict[str, TensorLocation]:
    """Build (or load a cached) tensor -> :class:`TensorLocation` index."""
    model_dir = Path(model_dir)
    sidecar = model_dir / _SIDECAR_FILENAME

    if use_cache and sidecar.exists():
        cached = json.loads(sidecar.read_text())
        if cached.get("version") == _SIDECAR_VERSION:
            return {
                name: TensorLocation(
                    shard=loc["shard"],
                    dtype=loc["dtype"],
                    shape=tuple(loc["shape"]),
                    offset=loc["offset"],
                    length=loc["length"],
                )
                for name, loc in cached["tensors"].items()
            }

    index: dict[str, TensorLocation] = {}
    for shard_name in _discover_shards(model_dir):
        index.update(_index_shard(model_dir / shard_name, shard_name))

    if use_cache:
        payload = {
            "version": _SIDECAR_VERSION,
            "tensors": {
                name: {
                    "shard": loc.shard,
                    "dtype": loc.dtype,
                    "shape": list(loc.shape),
                    "offset": loc.offset,
                    "length": loc.length,
                }
                for name, loc in index.items()
            },
        }
        sidecar.write_text(json.dumps(payload))

    return index
