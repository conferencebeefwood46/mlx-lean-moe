"""Per-architecture registry: parsing an HF ``config.json`` into a
family-specific config dataclass, and picking the model class for it.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class QuantScheme:
    """One MLX quantization scheme, including its packing mode."""

    bits: int
    group_size: int
    mode: str = "affine"

    def __post_init__(self) -> None:
        supported = {
            "affine": ({2, 3, 4, 5, 6, 8}, {32, 64, 128}),
            "mxfp4": ({4}, {32}),
            "mxfp8": ({8}, {32}),
            "nvfp4": ({4}, {16}),
        }
        if self.mode not in supported:
            raise ValueError(f"unsupported quantization mode {self.mode!r}")
        bits, groups = supported[self.mode]
        if self.bits not in bits or self.group_size not in groups:
            raise ValueError(
                f"unsupported {self.mode} quantization with bits={self.bits}, group_size={self.group_size}"
            )


class GenerativeModel(Protocol):
    """The surface :mod:`mlx_lean_moe.runtime.generate` needs from a
    model class."""

    def __call__(self, token_id: int) -> Any: ...

    def prefill(self, token_ids: list[int]) -> Any: ...

    def close(self) -> None: ...


_ADAPTERS: dict[str, Callable[[dict], Any]] = {}
_MODEL_CLASSES: dict[type, Callable[..., GenerativeModel]] = {}


def register_architecture(
    model_type: str,
    config_cls: type,
    adapter: Callable[[dict], Any],
    model_cls: Callable[..., GenerativeModel],
) -> None:
    """Register one architecture family. ``model_cls`` is looked up by the
    type of the parsed config, the only thing a later caller holds."""
    _ADAPTERS[model_type] = adapter
    _MODEL_CLASSES[config_cls] = model_cls


def model_config_from_hf(hf_config: dict) -> Any:
    """Build a family-specific config dataclass from a parsed HF
    ``config.json``, dispatching on its ``model_type``."""
    model_type = hf_config.get("model_type")
    adapter = _ADAPTERS.get(model_type)
    if adapter is None:
        known = ", ".join(sorted(_ADAPTERS))
        raise ValueError(
            f"no config adapter for model_type={model_type!r}; known: {known}"
        )
    return adapter(hf_config)


def model_class_for(config: Any) -> Callable[..., GenerativeModel]:
    """The model class that implements ``config``'s architecture, looked up
    by its concrete type."""
    model_cls = _MODEL_CLASSES.get(type(config))
    if model_cls is None:
        raise ValueError(
            f"no model class registered for config type {type(config).__name__}"
        )
    return model_cls
