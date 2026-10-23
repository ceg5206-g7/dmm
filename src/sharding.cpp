#include "dmm/sharding.hpp"
#include "dmm/bsp_plan.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace dmm {
std::size_t padded_size(std::size_t original, std::size_t P) {
    if (original == 0 || P == 0) throw std::invalid_argument("matrix size and P must be positive");
    const auto units = (original - 1) / P + 1;
    if (units > std::numeric_limits<std::size_t>::max() / P)
        throw std::invalid_argument("padded matrix size overflows");
    const auto n = units * P;
    detail::elements<float>(n);
    return n;
}

namespace {
void conversion(BspPlan& plan, std::size_t n, std::size_t c,
                          std::span<const std::size_t> rows, std::span<const std::size_t> blocks,
                          std::size_t tag, std::span<const std::size_t> scratch, bool input) {
    const auto P = plan.nodes(), q = grid_size(P,c);
    if (!n || n % P != 0 || rows.size() != P || blocks.size() != P || (!scratch.empty() && scratch.size() != P))
        throw std::invalid_argument("invalid prepared row/block conversion dimensions");
    detail::elements<float>(n);
    const auto height = n / P, block = n / q;
    for (std::size_t node = 0; node < P; ++node) {
        if (!plan.owns(node, rows[node]) || !plan.owns(node, blocks[node]) ||
            plan.buffer(rows[node]).size() < n * height ||
            plan.buffer(blocks[node]).size() < (node < q*q ? block*block : 0))
            throw std::invalid_argument("prepared conversion buffers have wrong owner/size");
        if (P > 1 && rows[node] == blocks[node] && node < q*q && node / q == node * height / block) {
            if (scratch.empty() || !plan.owns(node, scratch[node]) || scratch[node] == rows[node] ||
                plan.buffer(scratch[node]).size() < height * block)
                throw std::invalid_argument("aliased prepared conversion needs separate owned scratch");
        }
    }
    if (P == 1) {
        if (rows[0] != blocks[0]) throw std::invalid_argument("single-node conversion uses the same row/block buffer");
        return;
    }
    const auto send_step = plan.step(input ? "input_conversion" : "output_conversion");
    const auto recv_step = plan.step(input ? "input_conversion" : "output_conversion");
    // Retain every self rectangle before other receives overwrite aliased storage.
    for (std::size_t node = 0; node < q*q; ++node) if (node / q == node * height / block) {
        const auto i = node / q, j = node % q;
        const BufferView row{rows[node], j * block, n};
        const BufferView tile{blocks[node], (node * height - i * block) * block, block};
        if (rows[node] == blocks[node]) {
            const BufferView temp{scratch[node],0,block};
            plan.copy(recv_step,node,input ? row : tile,temp,height,block);
            plan.copy(recv_step,node,temp,input ? tile : row,height,block);
        } else plan.copy(recv_step,node,input ? row : tile,input ? tile : row,height,block);
    }
    for (std::size_t row_node = 0; row_node < P; ++row_node) {
        const auto i = row_node * height / block;
        for (std::size_t j = 0; j < q; ++j) {
            const auto tile_node = i*q+j;
            if (row_node == tile_node) continue;
            const BufferView row{rows[row_node],j*block,n};
            const BufferView tile{blocks[tile_node],(row_node*height-i*block)*block,block};
            const auto source = input ? row_node : tile_node, dest = input ? tile_node : row_node;
            const auto msg = plan.send(send_step,source,dest,tag,input ? row : tile,
                                       {row_node*height,j*block,height,block});
            plan.receive(recv_step,dest,msg,input ? tile : row);
        }
    }
}
}
void rows_to_blocks(BspPlan& plan, std::size_t n, std::size_t c,
                    std::span<const std::size_t> rows, std::span<const std::size_t> blocks,
                    std::size_t tag, std::span<const std::size_t> scratch) {
    conversion(plan,n,c,rows,blocks,tag,scratch,true);
}
void blocks_to_rows(BspPlan& plan, std::size_t n, std::size_t c,
                    std::span<const std::size_t> blocks, std::span<const std::size_t> rows,
                    std::size_t tag, std::span<const std::size_t> scratch) {
    conversion(plan,n,c,rows,blocks,tag,scratch,false);
}

} // namespace dmm
