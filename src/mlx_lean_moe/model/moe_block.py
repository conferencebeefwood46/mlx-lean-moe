"""Generic quantized-matmul helper, shared by every architecture's attention
and MoE code."""

from dataclasses import dataclass

import mlx.core as mx

from mlx_lean_moe.config import QuantScheme

QuantizedTensor = dict[str, mx.array]  # weight + scales, and optionally biases
LinearTensor = mx.array | QuantizedTensor


@dataclass(frozen=True, slots=True)
class LinearWeights:
    """A loaded linear projection with its own storage scheme; ``quant`` is
    ``None`` for an ordinary floating-point matrix."""

    tensors: LinearTensor
    quant: QuantScheme | None


def quantized_linear(
    x: mx.array,
    tensors: LinearWeights | LinearTensor,
    quant: QuantScheme | None = None,
) -> mx.array:
    """Linear projection for self-describing, legacy quantized, or dense weights."""

    if isinstance(tensors, LinearWeights):
        quant = tensors.quant
        tensors = tensors.tensors

    if isinstance(tensors, mx.array):
        # A weight-only projection is dense whatever the checkpoint-wide
        # fallback says; the on-disk fields are authoritative.
        return x @ tensors.T

    if quant is None:
        raise ValueError("quantized linear weights require a quantization scheme")

    return mx.quantized_matmul(
        x,
        tensors["weight"],
        scales=tensors["scales"],
        biases=tensors.get("biases"),
        group_size=quant.group_size,
        bits=quant.bits,
        mode=quant.mode,
    )
