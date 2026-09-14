"""qwen3_5's MoE block: a router, ``experts_per_token`` streamed routed
experts, and one always-on shared expert.

The router softmaxes over every expert before selecting, as the reference
does.
"""

from collections.abc import Callable

import mlx.core as mx
import mlx.nn as nn

from mlx_lean_moe.model.moe_block import (
    LinearWeights,
    QuantizedTensor,
    quantized_linear,
)
from mlx_lean_moe.model.qwen3_5.config import Qwen3_5Config
from mlx_lean_moe.weights.expert_loader import CachedExpertLoader


def _swiglu(gate: mx.array, up: mx.array) -> mx.array:
    return nn.silu(gate) * up


def swiglu_mlp(
    x: mx.array, tensors: dict[str, LinearWeights | QuantizedTensor], quant=None
) -> mx.array:
    """``x``: ``(hidden_size,)`` or ``(L, hidden_size)``; returns the same
    rank. Used for the shared expert."""
    squeeze = x.ndim == 1
    rows = x[None, :] if squeeze else x
    gate = quantized_linear(rows, tensors["gate_proj"], quant)
    up = quantized_linear(rows, tensors["up_proj"], quant)
    out = quantized_linear(_swiglu(gate, up), tensors["down_proj"], quant)
    return out[0] if squeeze else out


class Qwen3_5Router:
    def __init__(self, config: Qwen3_5Config, gate: QuantizedTensor) -> None:
        self.config = config
        self.gate = gate

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """``x``: ``(hidden_size,)`` or ``(L, hidden_size)``. Returns
        ``(top_k_indices, top_k_weights)`` with a matching leading shape."""
        squeeze = x.ndim == 1
        rows = x[None, :] if squeeze else x

        logits = quantized_linear(rows, self.gate, self.config.router_quant)
        gates = mx.softmax(logits, axis=-1, precise=True)

        k = self.config.experts_per_token
        indices = mx.argpartition(-gates, kth=k - 1, axis=-1)[..., :k]
        scores = mx.take_along_axis(gates, indices, axis=-1)
        if self.config.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)

        if squeeze:
            return indices[0], scores[0]
        return indices, scores


class Qwen3_5Experts:
    """The routed experts, streamed on demand: the same weighted sum of
    per-expert SwiGLU MLPs the reference computes over a resident tensor."""

    def __init__(
        self, config: Qwen3_5Config, layer_index: int, expert_loader: CachedExpertLoader
    ) -> None:
        self.config = config
        self.layer_index = layer_index
        self.expert_loader = expert_loader
        prefix = f"{config.tensor_prefix}.layers.{layer_index}.mlp.switch_mlp"
        # Resolved once: a mixed checkpoint can publish hundreds of overrides.
        self.projection_quant = {
            proj: config.quant_for(f"{prefix}.{proj}", config.expert_quant)
            for proj in ("gate_proj", "up_proj", "down_proj")
        }

    def __call__(
        self,
        x: mx.array,
        indices: mx.array,
        weights: mx.array,
        *,
        before_load: Callable[[], None] | None = None,
    ) -> mx.array:
        """``x``: ``(hidden_size,)`` or ``(L, hidden_size)``, returned at the
        same rank. ``before_load`` runs between routing and the reads."""
        squeeze = x.ndim == 1
        rows = x[None, :] if squeeze else x
        picks = indices[None, :] if squeeze else indices
        scores = weights[None, :] if squeeze else weights
        num_rows, k = picks.shape

        slots_by_expert: dict[int, list[int]] = {}
        for position, chosen in enumerate(picks.tolist()):
            for slot, expert in enumerate(chosen):
                slots_by_expert.setdefault(expert, []).append(position * k + slot)

        if before_load is not None:
            before_load()

        loaded = self.expert_loader.load_many(self.layer_index, list(slots_by_expert))
        grouped = [
            self._through_expert(tensors, rows[mx.array([slot // k for slot in slots])])
            for tensors, slots in zip(loaded, slots_by_expert.values())
        ]

        order = [slot for slots in slots_by_expert.values() for slot in slots]
        inverse = [0] * len(order)
        for source, destination in enumerate(order):
            inverse[destination] = source
        combined = mx.concatenate(grouped, axis=0)[mx.array(inverse)]

        out = (scores[..., None] * combined.reshape(num_rows, k, -1)).sum(axis=1)
        return out[0] if squeeze else out

    def _through_expert(self, tensors: dict, taken: mx.array) -> mx.array:
        gate = quantized_linear(
            taken, tensors["gate_proj"], self.projection_quant["gate_proj"]
        )
        up = quantized_linear(
            taken, tensors["up_proj"], self.projection_quant["up_proj"]
        )
        return quantized_linear(
            _swiglu(gate, up), tensors["down_proj"], self.projection_quant["down_proj"]
        )


class Qwen3_5SharedExpert:
    """One wide expert every token goes through, scaled by its own learned
    sigmoid gate. Resident rather than streamed."""

    def __init__(
        self,
        config: Qwen3_5Config,
        projections: dict[str, LinearWeights | QuantizedTensor],
        gate: LinearWeights | QuantizedTensor,
    ) -> None:
        self.config = config
        self.projections = projections
        self.gate = gate

    def __call__(self, x: mx.array) -> mx.array:
        out = swiglu_mlp(x, self.projections, self.config.other_quant)
        squeeze = x.ndim == 1
        rows = x[None, :] if squeeze else x
        gate_logit = quantized_linear(rows, self.gate, self.config.shared_gate_quant)
        gate = mx.sigmoid(gate_logit[0] if squeeze else gate_logit)
        return gate * out
