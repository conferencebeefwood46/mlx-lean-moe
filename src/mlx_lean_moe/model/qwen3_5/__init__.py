"""The qwen3_5 (Qwen3.5-MoE) architecture family.

One layer in every ``full_attention_interval`` uses softmax attention; the
rest are gated-delta linear attention, whose state is fixed-size.
"""

from mlx_lean_moe.config import register_architecture
from mlx_lean_moe.model.qwen3_5.config import Qwen3_5Config, from_hf_config
from mlx_lean_moe.model.qwen3_5.transformer import Qwen3_5Transformer

register_architecture(
    model_type="qwen3_5_moe",
    config_cls=Qwen3_5Config,
    adapter=from_hf_config,
    model_cls=Qwen3_5Transformer,
)
