"""On-demand reads of one expert's slice out of a *stacked* MoE tensor.

All of one projection's experts live in a single tensor with experts along
axis 0, which is C-contiguous, so one expert is a contiguous byte range.
"""

import os
from concurrent.futures import Executor, as_completed
from pathlib import Path

import mlx.core as mx

from mlx_lean_moe.weights._shard_fds import ShardFdCache
from mlx_lean_moe.weights.expert_loader import DEFAULT_PROJECTIONS, decode_tensor, keeps_stored_width
from mlx_lean_moe.weights.safetensors_index import TensorLocation

_FIELDS = ("weight", "scales", "biases")


class StackedExpertStreamer:
    """Reads individual experts on demand via ``os.pread``, slicing them out
    of a stacked ``(num_experts, ...)`` on-disk tensor."""

    def __init__(
        self,
        model_dir: str | Path,
        index: dict[str, TensorLocation],
        num_experts: int,
        layer_prefix: str = "language_model.model.layers",
        projections: tuple[str, ...] = DEFAULT_PROJECTIONS,
        stack_name: str = "experts.switch_glu",
    ) -> None:
        self.model_dir = Path(model_dir)
        self.index = index
        self.num_experts = num_experts
        self.layer_prefix = layer_prefix
        self.projections = projections
        # What the stacked-expert module is called between the layer index
        # and the projection name.
        self.stack_name = stack_name
        self._fds = ShardFdCache(self.model_dir)

    def _tensor_name(self, layer: int, proj: str, field: str) -> str:
        return f"{self.layer_prefix}.{layer}.{self.stack_name}.{proj}.{field}"

    def _expert_location(self, whole: TensorLocation, expert: int) -> TensorLocation:
        """`expert`'s contiguous byte range within the stacked tensor
        `whole`, which is C-contiguous and stacked along axis 0."""
        per_expert_length = whole.length // self.num_experts
        return TensorLocation(
            shard=whole.shard,
            dtype=whole.dtype,
            shape=whole.shape[1:],
            offset=whole.offset + expert * per_expert_length,
            length=per_expert_length,
        )

    def read_expert_tensor(self, layer: int, expert: int, proj: str, field: str) -> mx.array:
        whole = self.index[self._tensor_name(layer, proj, field)]
        loc = self._expert_location(whole, expert)
        fd = self._fds.fd_for_shard(loc.shard)
        raw = os.pread(fd, loc.length, loc.offset)
        return decode_tensor(raw, loc.dtype, loc.shape, stored_width=keeps_stored_width(field))

    def _projection_fields(self, layer: int, proj: str) -> tuple[str, ...]:
        """Which of weight/scales/biases this projection actually has: a
        mixed checkpoint can leave one dense while its neighbours are not."""
        fields = tuple(field for field in _FIELDS if self._tensor_name(layer, proj, field) in self.index)
        if "weight" not in fields:
            raise KeyError(f"no weight tensor for expert projection layer {layer} {proj}")
        if "scales" not in fields and fields != ("weight",):
            raise ValueError(f"expert projection layer {layer} {proj} has biases but no scales")
        return fields

    @staticmethod
    def _assemble_projection(tensors: dict[str, mx.array]):
        """A weight-only projection is dense, and is handed back as the array
        itself rather than a one-key dict."""
        return tensors["weight"] if tensors.keys() == {"weight"} else tensors

    def load_expert(self, layer: int, expert: int) -> dict[str, dict[str, mx.array]]:
        return {
            proj: self._assemble_projection(
                {
                    field: self.read_expert_tensor(layer, expert, proj, field)
                    for field in self._projection_fields(layer, proj)
                }
            )
            for proj in self.projections
        }

    def load_experts_concurrently(
        self, layer: int, experts: list[int], executor: Executor
    ) -> list[dict[str, dict[str, mx.array]]]:
        """`load_expert` for each of `experts`, every read submitted to
        `executor` as one flat batch: nested futures could deadlock."""
        projection_fields = {proj: self._projection_fields(layer, proj) for proj in self.projections}
        keys = [
            (expert, proj, field)
            for expert in experts
            for proj in self.projections
            for field in projection_fields[proj]
        ]
        futures = {
            executor.submit(self.read_expert_tensor, layer, expert, proj, field): (expert, proj, field)
            for expert, proj, field in keys
        }
        partial: dict[int, dict[str, dict[str, mx.array]]] = {}
        for future in as_completed(futures):
            expert, proj, field = futures[future]
            partial.setdefault(expert, {}).setdefault(proj, {})[field] = future.result()
        return [
            {proj: self._assemble_projection(partial[expert][proj]) for proj in self.projections} for expert in experts
        ]

    def close(self) -> None:
        self._fds.close()
