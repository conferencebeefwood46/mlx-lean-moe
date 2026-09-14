"""One routed expert as one contiguous read.

A stacked checkpoint scatters an expert across nine byte ranges; reading it
costs nine ``pread`` calls, and read count is what the wait follows. This
repacks them into one. Same bytes, different order -- :func:`verify` checks
that against the original. Delete the pack and the engine reads as before.
"""

import json
import os
from collections.abc import Sequence
from concurrent.futures import Executor, as_completed
from pathlib import Path

import mlx.core as mx

from mlx_lean_moe.weights._shard_fds import ShardFdCache
from mlx_lean_moe.weights.expert_loader import (
    DEFAULT_PROJECTIONS,
    decode_tensor,
    keeps_stored_width,
)
from mlx_lean_moe.weights.safetensors_index import TensorLocation

PACK_NAME = "experts.pack"
INDEX_NAME = "experts.pack.json"
VERSION = 1


class PackLayout:
    """Where every expert's bytes live, and how to cut one back into tensors."""

    def __init__(self, document: dict) -> None:
        if document.get("version") != VERSION:
            raise ValueError(
                f"expert pack version {document.get('version')!r}, expected {VERSION}"
            )
        self.num_experts: int = document["num_experts"]
        self.projections: tuple[str, ...] = tuple(document["projections"])
        self.layers: dict[int, dict] = {
            int(k): v for k, v in document["layers"].items()
        }

    def expert_range(self, layer: int, expert: int) -> tuple[int, int]:
        entry = self.layers[layer]
        if not 0 <= expert < self.num_experts:
            raise IndexError(f"expert {expert} is outside 0..{self.num_experts - 1}")
        return entry["offset"] + expert * entry["stride"], entry["stride"]

    def fields(self, layer: int) -> list[dict]:
        return self.layers[layer]["fields"]


def _field_locations(
    index: dict[str, TensorLocation],
    layer_prefix: str,
    stack_name: str,
    projections: Sequence[str],
    layer: int,
) -> list[tuple[str, str, TensorLocation]]:
    """The (projection, field, location) triples a layer actually has."""
    found = []
    for projection in projections:
        for field in ("weight", "scales", "biases"):
            name = f"{layer_prefix}.{layer}.{stack_name}.{projection}.{field}"
            if name in index:
                found.append((projection, field, index[name]))
    return found


def build(
    model_dir: str | Path,
    index: dict[str, TensorLocation],
    num_experts: int,
    num_layers: int,
    layer_prefix: str,
    stack_name: str,
    projections: Sequence[str] = DEFAULT_PROJECTIONS,
    *,
    progress=None,
) -> Path:
    """Write a pack beside the checkpoint. Returns its path."""
    model_dir = Path(model_dir)
    fds = ShardFdCache(model_dir)
    layers: dict[str, dict] = {}
    written = 0

    try:
        with open(model_dir / PACK_NAME, "wb") as pack:
            for layer in range(num_layers):
                found = _field_locations(
                    index, layer_prefix, stack_name, projections, layer
                )
                if not found:
                    raise KeyError(f"no stacked expert tensors for layer {layer}")

                descriptors = []
                within = 0
                for projection, field, loc in found:
                    length = loc.length // num_experts
                    if length * num_experts != loc.length:
                        raise ValueError(
                            f"{layer_prefix}.{layer}.{stack_name}.{projection}.{field} is {loc.length} "
                            f"bytes over {num_experts} experts, which does not divide"
                        )
                    descriptors.append(
                        {
                            "projection": projection,
                            "field": field,
                            "dtype": loc.dtype,
                            "shape": list(loc.shape[1:]),
                            "offset": within,
                            "length": length,
                        }
                    )
                    within += length

                layers[str(layer)] = {
                    "offset": written,
                    "stride": within,
                    "fields": descriptors,
                }

                # Expert-major: everything one expert needs, then the next.
                for expert in range(num_experts):
                    for (_, _, loc), descriptor in zip(found, descriptors):
                        length = descriptor["length"]
                        fd = fds.fd_for_shard(loc.shard)
                        pack.write(os.pread(fd, length, loc.offset + expert * length))
                    written += within

                if progress is not None:
                    progress(layer + 1, num_layers)
    finally:
        fds.close()

    document = {
        "version": VERSION,
        "num_experts": num_experts,
        "projections": list(projections),
        "layers": layers,
    }
    (model_dir / INDEX_NAME).write_text(json.dumps(document))
    return model_dir / PACK_NAME


def verify(packed, original, layers: Sequence[int], experts: Sequence[int]) -> None:
    """Check a pack against the checkpoint it was built from, raising on the
    first disagreement."""
    for layer in layers:
        for expert in experts:
            want = original.load_expert(layer, expert)
            got = packed.load_expert(layer, expert)
            if got.keys() != want.keys():
                raise ValueError(
                    f"layer {layer} expert {expert}: projections {sorted(got)} != {sorted(want)}"
                )
            for projection, reference in want.items():
                mine = got[projection]
                pairs = (
                    [("", mine, reference)]
                    if isinstance(reference, mx.array)
                    else [
                        (field, mine[field], tensor)
                        for field, tensor in reference.items()
                    ]
                )
                for field, a, b in pairs:
                    if a.dtype != b.dtype or a.shape != b.shape:
                        raise ValueError(
                            f"layer {layer} expert {expert} {projection}.{field}: "
                            f"{a.dtype}{a.shape} != {b.dtype}{b.shape}"
                        )
                    if not mx.array_equal(a, b).item():
                        raise ValueError(
                            f"layer {layer} expert {expert} {projection}.{field}: values differ"
                        )


def load_layout(model_dir: str | Path) -> PackLayout | None:
    """The pack's layout, or None when the checkpoint has no pack."""
    model_dir = Path(model_dir)
    document = model_dir / INDEX_NAME
    if not document.exists() or not (model_dir / PACK_NAME).exists():
        return None
    return PackLayout(json.loads(document.read_text()))


class ExpertPackStreamer:
    """Reads whole experts out of a pack, one ``pread`` each. Duck-typed
    against :class:`StackedExpertStreamer`."""

    def __init__(self, model_dir: str | Path, layout: PackLayout) -> None:
        self.model_dir = Path(model_dir)
        self.layout = layout
        self.projections = layout.projections
        self.num_experts = layout.num_experts
        self._fd = os.open(self.model_dir / PACK_NAME, os.O_RDONLY)

    def read_expert(
        self, layer: int, expert: int
    ) -> dict[str, dict[str, mx.array] | mx.array]:
        offset, stride = self.layout.expert_range(layer, expert)
        blob = os.pread(self._fd, stride, offset)
        if len(blob) != stride:
            raise OSError(
                f"pack read returned {len(blob)} bytes for expert {expert} of layer {layer}, want {stride}"
            )

        gathered: dict[str, dict[str, mx.array]] = {}
        for descriptor in self.layout.fields(layer):
            start = descriptor["offset"]
            raw = blob[start : start + descriptor["length"]]
            field = descriptor["field"]
            gathered.setdefault(descriptor["projection"], {})[field] = decode_tensor(
                raw,
                descriptor["dtype"],
                tuple(descriptor["shape"]),
                stored_width=keeps_stored_width(field),
            )
        return {
            projection: (tensors["weight"] if tensors.keys() == {"weight"} else tensors)
            for projection, tensors in gathered.items()
        }

    # The names CachedExpertLoader calls.
    def load_expert(self, layer: int, expert: int):
        return self.read_expert(layer, expert)

    def load_experts_concurrently(
        self, layer: int, experts: list[int], executor: Executor
    ):
        """One read per expert rather than one per field, so the fan-out is
        over experts. Results come back in the order asked for."""
        futures = {
            executor.submit(self.read_expert, layer, expert): expert
            for expert in experts
        }
        done: dict[int, dict] = {}
        for future in as_completed(futures):
            done[futures[future]] = future.result()
        return [done[expert] for expert in experts]

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


def build_for_checkpoint(model_dir: str | Path, *, progress=None) -> Path:
    """Build a pack from a checkpoint's own config."""
    from mlx_lean_moe.config import model_config_from_hf
    from mlx_lean_moe.weights.safetensors_index import build_index

    model_dir = Path(model_dir)
    config = model_config_from_hf(json.loads((model_dir / "config.json").read_text()))
    return build(
        model_dir,
        build_index(model_dir),
        num_experts=config.num_experts,
        num_layers=config.num_layers,
        layer_prefix=f"{config.tensor_prefix}.layers",
        stack_name="mlp.switch_mlp",
        progress=progress,
    )


def checkpoint_dir(checkpoint: str | Path) -> Path:
    """Resolve a hub repo id, or a directory, to the checkpoint's directory.
    A repo id must already be downloaded."""
    path = Path(checkpoint)
    if path.is_dir():
        return path
    if path.exists() or path.parts[:1] in ((".",), ("..",), ("/",)):
        raise NotADirectoryError(f"{checkpoint} is not a checkpoint directory")

    from mlx_lean_moe.weights import download

    cached = download.snapshot_dir(str(checkpoint))
    if cached is None:
        raise FileNotFoundError(
            f"{checkpoint} is neither a directory nor a downloaded repo id; fetch it with\n"
            f"  python -m mlx_lean_moe.weights.download {checkpoint}"
        )
    return cached


def _main() -> None:
    import argparse
    import sys
    import time

    parser = argparse.ArgumentParser(
        description="Repack a checkpoint's routed experts so each is one contiguous read. "
        "Writes experts.pack beside the weights; delete it to go back."
    )
    parser.add_argument(
        "checkpoint", help="a downloaded hub repo id, or a checkpoint directory"
    )
    arguments = parser.parse_args()

    try:
        model_dir = checkpoint_dir(arguments.checkpoint)
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise SystemExit(str(exc)) from exc

    started = time.time()

    def show(done: int, total: int) -> None:
        print(
            f"  layer {done}/{total}, {time.time() - started:.0f}s",
            end="\r",
            file=sys.stderr,
        )

    path = build_for_checkpoint(model_dir, progress=show)
    print(
        f"\nwrote {path} ({path.stat().st_size / 2**30:.2f} GiB) in {time.time() - started:.0f}s"
    )


if __name__ == "__main__":
    _main()
