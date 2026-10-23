#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>

namespace dmm {

inline constexpr std::size_t GEMM_TILE = 32;

namespace detail {
inline void check_gemm_storage(std::size_t rows, std::size_t cols, std::size_t stride, std::size_t words) {
    if (rows == 0 || cols == 0)
        return;
    if (stride < cols || rows - 1 > (std::numeric_limits<std::size_t>::max() - cols) / stride ||
        (rows - 1) * stride + cols > words)
        throw std::invalid_argument("GEMM storage/stride does not cover the requested rectangle");
}
} // namespace detail

// C += A*B. Row-major FP32 rectangles; C must not overlap A/B. No allocation.
// Every layout uses this tile and compilation flags; k stays in ascending order.
inline void gemm_add(std::size_t m, std::size_t n, std::size_t k, std::span<const float> a, std::size_t lda,
                     std::span<const float> b, std::size_t ldb, std::span<float> c, std::size_t ldc) {
    detail::check_gemm_storage(m, k, lda, a.size());
    detail::check_gemm_storage(k, n, ldb, b.size());
    detail::check_gemm_storage(m, n, ldc, c.size());
    if (m == 0 || n == 0 || k == 0)
        return;
    for (std::size_t ii = 0; ii < m; ii += GEMM_TILE)
        for (std::size_t kk = 0; kk < k; kk += GEMM_TILE)
            for (std::size_t jj = 0; jj < n; jj += GEMM_TILE)
                for (std::size_t i = ii; i < std::min(m, ii + GEMM_TILE); ++i)
                    for (std::size_t x = kk; x < std::min(k, kk + GEMM_TILE); ++x) {
                        const auto value = a[i * lda + x];
                        for (std::size_t j = jj; j < std::min(n, jj + GEMM_TILE); ++j)
                            c[i * ldc + j] += value * b[x * ldb + j];
                    }
}

} // namespace dmm
