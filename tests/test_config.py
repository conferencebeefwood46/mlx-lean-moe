import pytest

from mlx_lean_moe.config import model_config_from_hf


def test_unknown_model_type_raises():
    with pytest.raises(ValueError, match="no config adapter"):
        model_config_from_hf({"model_type": "some_future_arch"})
