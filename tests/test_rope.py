"""`apply_leading_rope` rotates only the leading `rotary_dim` dimensions;
see `model/rope.py` for the other partial-rotary convention.
"""

import mlx.core as mx

from mlx_lean_moe.model.rope import apply_leading_rope

HEAD_DIM = 16


def test_leading_rope_leaves_the_untouched_tail_bit_identical():

    head_dim, rotary_dim = 16, 4
    mx.random.seed(1)
    x = mx.random.normal((1, 2, 3, head_dim))
    out = apply_leading_rope(x, rotary_dim, 10000.0, offset=7)
    mx.eval(out)
    assert mx.array_equal(out[..., rotary_dim:], x[..., rotary_dim:]).item()
