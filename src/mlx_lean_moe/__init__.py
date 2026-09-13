"""Importing this package registers every supported architecture with
``config.py``'s dispatch, whichever submodule a caller imports first."""

from mlx_lean_moe import model  # noqa: F401
