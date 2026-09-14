"""Every module has to import on its own.

Annotations are evaluated as they are read, so a class named before it is
defined raises `NameError` at import rather than at use.
"""

import importlib
import pkgutil

import pytest

import mlx_lean_moe

MODULES = sorted(
    module.name
    for module in pkgutil.walk_packages(mlx_lean_moe.__path__, "mlx_lean_moe.")
)


def test_the_package_has_modules_to_check():
    """A walk that found nothing would pass the test below silently."""
    assert len(MODULES) > 20


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)
