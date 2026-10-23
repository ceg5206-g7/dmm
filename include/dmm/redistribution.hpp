#pragma once
#include "dmm/matmul.hpp"
#include <cmath>

namespace dmm {
// Grid contract for 2D/2.5D: P = c*q*q and c divides q.
inline std::size_t grid_size(std::size_t P, std::size_t c = 1) {
    if (P == 0 || c == 0 || P % c != 0)
        throw std::invalid_argument("P and c must be positive and c must divide P");
    const auto plane = P / c;
    const auto q = static_cast<std::size_t>(std::sqrt(static_cast<long double>(plane)));
    if (q == 0 || plane / q != q || plane % q != 0 || q % c != 0)
        throw std::invalid_argument("grid requires P = c*q*q and c must divide q");
    return q;
}

std::size_t padded_size(std::size_t original_size, std::size_t P);
void add_rows_to_tiles(BspPlan&, const MatmulConfig&, std::span<const MatrixView> rows,
                       std::span<const MatrixView> tiles, std::span<const MatrixView> scratch = {});
void add_tiles_to_rows(BspPlan&, const MatmulConfig&, std::span<const MatrixView> tiles,
                       std::span<const MatrixView> rows, std::span<const MatrixView> scratch = {});
} // namespace dmm
