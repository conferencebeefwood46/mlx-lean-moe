import pytest

from mlx_lean_moe.config import model_config_from_hf

# The validation checkpoint's own text_config, trimmed to 8 layers and the
# fields the adapter reads.
QWEN3_5_TEXT_CONFIG = {
    "model_type": "qwen3_5_moe_text",
    "hidden_size": 2048,
    "vocab_size": 248320,
    "num_hidden_layers": 8,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "rms_norm_eps": 1e-06,
    "full_attention_interval": 4,
    "layer_types": [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ],
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "rope_parameters": {
        "mrope_interleaved": True,
        "mrope_section": [11, 11, 10],
        "partial_rotary_factor": 0.25,
        "rope_theta": 10000000,
        "rope_type": "default",
    },
    "num_experts": 256,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 512,
    "shared_expert_intermediate_size": 512,
    "mlp_only_layers": [],
    "decoder_sparse_step": 1,
    # The checkpoint really does declare this while shipping no MTP tensors;
    # the adapter must not treat it as an error.
    "mtp_num_hidden_layers": 1,
    "tie_word_embeddings": False,
}

QWEN3_5_QUANTIZATION = {
    "group_size": 64,
    "bits": 4,
    "mode": "affine",
    "language_model.model.layers.0.mlp.gate": {"group_size": 64, "bits": 8},
    "language_model.model.layers.0.mlp.shared_expert_gate": {
        "group_size": 64,
        "bits": 8,
    },
}

QWEN3_5_CONFIG = {
    "model_type": "qwen3_5_moe",
    "text_config": QWEN3_5_TEXT_CONFIG,
    "quantization_config": QWEN3_5_QUANTIZATION,
}


def test_qwen3_5_maps_core_fields():
    config = model_config_from_hf(QWEN3_5_CONFIG)
    assert config.num_layers == 8
    assert config.hidden_size == 2048
    assert config.vocab_size == 248320
    assert config.num_experts == 256
    assert config.experts_per_token == 8
    assert config.moe_intermediate_size == 512
    assert config.shared_expert_intermediate_size == 512


def test_qwen3_5_marks_three_in_four_layers_as_linear():
    config = model_config_from_hf(QWEN3_5_CONFIG)
    assert config.is_linear_per_layer == (
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        False,
    )


def test_qwen3_5_derives_layer_types_when_not_published():
    text = dict(QWEN3_5_TEXT_CONFIG)
    del text["layer_types"]

    config = model_config_from_hf(
        {
            "model_type": "qwen3_5_moe",
            "text_config": text,
            "quantization_config": QWEN3_5_QUANTIZATION,
        }
    )
    # Same pattern derived from full_attention_interval=4.
    assert config.is_linear_per_layer == (
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        False,
    )


def test_qwen3_5_linear_attention_shape():
    config = model_config_from_hf(QWEN3_5_CONFIG)
    linear = config.linear
    assert linear.num_key_heads == 16
    assert linear.num_value_heads == 32
    assert linear.key_dim == 16 * 128
    assert linear.value_dim == 32 * 128
    # The short causal convolution runs over q, k and v concatenated.
    assert linear.conv_dim == 2 * (16 * 128) + 32 * 128
    assert linear.conv_kernel_dim == 4


def test_qwen3_5_partial_rotary_and_rope_theta():
    config = model_config_from_hf(QWEN3_5_CONFIG)
    assert config.head_dim == 256
    assert config.rotary_dim == 64  # 256 * 0.25
    assert config.rope_theta == 10000000


def test_qwen3_5_quant_overrides_are_read_independently():
    config = model_config_from_hf(QWEN3_5_CONFIG)
    assert config.expert_quant.bits == 4
    assert config.expert_quant.group_size == 64
    assert config.router_quant.bits == 8
    assert config.shared_gate_quant.bits == 8
    assert config.other_quant.bits == 4


def test_qwen3_5_resolves_exact_mixed_quantization_per_projection():
    quant = dict(QWEN3_5_QUANTIZATION)
    quant.update(
        {
            "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight": {
                "bits": 3,
                "group_size": 32,
            },
            "language_model.model.layers.0.mlp.switch_mlp.up_proj": {"bits": 6},
            "language_model.model.layers.1.mlp.switch_mlp.gate_proj": {
                "bits": 8,
                "group_size": 32,
                "mode": "mxfp8",
            },
        }
    )

    config = model_config_from_hf(
        {
            "model_type": "qwen3_5_moe",
            "text_config": QWEN3_5_TEXT_CONFIG,
            "quantization_config": quant,
        }
    )

    layer0 = "language_model.model.layers.0.mlp.switch_mlp"
    layer1 = "language_model.model.layers.1.mlp.switch_mlp"
    assert config.quant_for(f"{layer0}.gate_proj").bits == 3
    assert config.quant_for(f"{layer0}.gate_proj").group_size == 32
    # Missing fields inherit the checkpoint-wide default.
    assert config.quant_for(f"{layer0}.up_proj").group_size == 64
    # Similar suffixes in adjacent layers must not bleed into each other.
    assert config.quant_for(f"{layer1}.gate_proj").bits == 8
    assert config.quant_for(f"{layer1}.gate_proj").mode == "mxfp8"
    assert config.quant_for(f"{layer1}.up_proj") == config.other_quant


def test_qwen3_5_norm_topk_prob_defaults_to_true_when_absent():
    """The reference implementation's own default, and it changes the routed
    scores."""

    config = model_config_from_hf(QWEN3_5_CONFIG)
    assert config.norm_topk_prob is True


def test_qwen3_5_ignores_a_multi_token_prediction_head():
    """The head sits outside the num_hidden_layers stack and plain
    generation never runs it, so it must not read as unsupported."""

    text = dict(QWEN3_5_TEXT_CONFIG)
    text["mtp_num_hidden_layers"] = 1
    config = model_config_from_hf(
        {
            "model_type": "qwen3_5_moe",
            "text_config": text,
            "quantization_config": QWEN3_5_QUANTIZATION,
        }
    )
    assert config.num_layers == 8  # unchanged: the MTP head isn't one of them


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("mlp_only_layers", [0, 1], "mlp_only_layers"),
        ("decoder_sparse_step", 2, "decoder_sparse_step"),
    ],
)
def test_qwen3_5_rejects_unimplemented_mechanisms(field, value, message):
    text = dict(QWEN3_5_TEXT_CONFIG)
    text[field] = value

    with pytest.raises(ValueError, match=message):
        model_config_from_hf(
            {
                "model_type": "qwen3_5_moe",
                "text_config": text,
                "quantization_config": QWEN3_5_QUANTIZATION,
            }
        )
