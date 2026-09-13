"""`python -m mlx_lean_moe ...`, the same entry point as the installed
`mlx-lean-moe` script."""

if __name__ == "__main__":
    from .server import _main

    _main()
