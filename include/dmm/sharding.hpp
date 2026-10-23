#pragma once
#include "dmm/bsp.hpp"
#include "dmm/layout.hpp"

namespace dmm {

std::size_t padded_size(std::size_t original_size, std::size_t P);

class BspPlan;
// Append common conversions to a plan. IDs name node-owned buffers;
// base-layer block buffers need n²/q² words, rows n²/P. Aliased rows/blocks need
// one n²/(P*q)-word scratch per node. P=1 uses the same row/block buffer.
void rows_to_blocks(BspPlan&, std::size_t n, std::size_t c,
                    std::span<const std::size_t> rows, std::span<const std::size_t> blocks,
                    std::size_t tag, std::span<const std::size_t> scratch = {});
void blocks_to_rows(BspPlan&, std::size_t n, std::size_t c,
                    std::span<const std::size_t> blocks, std::span<const std::size_t> rows,
                    std::size_t tag, std::span<const std::size_t> scratch = {});

} // namespace dmm
