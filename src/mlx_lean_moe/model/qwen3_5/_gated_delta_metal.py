"""Single-sequence, float32 gated-delta scan for Metal.

Adapted from mlx-lm 0.31.3, mlx_lm/models/gated_delta.py. Each SIMD group
owns one value row; its 32 lanes keep the key dimension in registers. The
input state is read-only, so callers can retain it across subsequent scans.

MIT License

Copyright © 2023 Apple Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from functools import lru_cache

import mlx.core as mx


@lru_cache(maxsize=1)
def _kernel():
    # Construct lazily: importing the package or running on CPU should not
    # require a Metal kernel. MLX caches compilation per template signature.
    return mx.fast.metal_kernel(
        name="lean_moe_gated_delta_scan_f32",
        input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
        output_names=["y", "state_out"],
        ensure_row_contiguous=True,
        source="""
            const uint hv = thread_position_in_grid.z;
            const uint dv = thread_position_in_grid.y;
            if (hv >= Hv || dv >= Dv) return;
            const uint hk = hv / (Hv / Hk);
            const uint lane = thread_position_in_threadgroup.x;
            constexpr int N = Dk / 32;

            auto q_ = q + hk * Dk;
            auto k_ = k + hk * Dk;
            auto v_ = v + hv * Dv + dv;
            auto y_ = y + hv * Dv + dv;
            auto g_ = g + hv;
            auto beta_ = beta + hv;
            const uint offset = (hv * Dv + dv) * Dk;

            float state[N];
            for (int i = 0; i < N; ++i) {
                state[i] = state_in[offset + N * lane + i];
            }
            for (int t = 0; t < T; ++t) {
                float read = 0.0f;
                for (int i = 0; i < N; ++i) {
                    const uint dk = N * lane + i;
                    state[i] = state[i] * g_[0];
                    read += state[i] * k_[dk];
                }
                read = simd_sum(read);
                const float delta = (v_[0] - read) * beta_[0];
                float out = 0.0f;
                for (int i = 0; i < N; ++i) {
                    const uint dk = N * lane + i;
                    state[i] = state[i] + k_[dk] * delta;
                    out += state[i] * q_[dk];
                }
                out = simd_sum(out);
                if (thread_index_in_simdgroup == 0) y_[0] = out;
                q_ += Hk * Dk;
                k_ += Hk * Dk;
                v_ += Hv * Dv;
                y_ += Hv * Dv;
                g_ += Hv;
                beta_ += Hv;
            }
            for (int i = 0; i < N; ++i) {
                state_out[offset + N * lane + i] = state[i];
            }
        """,
    )


def gated_delta_scan_metal(q, k, v, g, beta, state) -> tuple[mx.array, mx.array]:
    """Called only after the public scan has checked shapes and dtypes."""
    length, key_heads, key_dim = q.shape
    _, value_heads, value_dim = v.shape
    y, new_state = _kernel()(
        inputs=[q, k, v, g, beta, state, length],
        template=[("Dk", key_dim), ("Dv", value_dim), ("Hk", key_heads), ("Hv", value_heads)],
        grid=(32, value_dim, value_heads),
        threadgroup=(32, 4, 1),
        output_shapes=[v.shape, state.shape],
        output_dtypes=[mx.float32, mx.float32],
    )
    return y, new_state
