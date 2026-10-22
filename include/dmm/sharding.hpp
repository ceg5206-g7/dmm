#pragma once
#include "dmm/bsp.hpp"
#include "dmm/layout.hpp"

namespace dmm {

struct RowShards {
    std::size_t original_size, size;
    NodeBuffers values;
};
struct BlockShards {
    std::size_t original_size, size, c;
    NodeBuffers values; // Base layer first; other layers empty until replicated.
};

std::size_t padded_size(std::size_t original_size, std::size_t P);
RowShards make_row_shards(Bsp&, const Matrix<float>&);
Matrix<float> gather_row_shards(const RowShards&);
BlockShards rows_to_blocks(Bsp&, Executor&, const RowShards&, std::size_t c, std::size_t tag);
RowShards blocks_to_rows(Bsp&, Executor&, const BlockShards&, std::size_t tag);

} // namespace dmm
