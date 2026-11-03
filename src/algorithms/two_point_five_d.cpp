#include "dmm/algorithms.hpp"
#include "dmm/collectives.hpp"
#include "dmm/redistribution.hpp"

namespace dmm {
// 2.5D: P = c*q*q nodes form c layers of q*q grids, node (layer, i, j) living at
// layer*q*q + i*q + j. Row shards convert to base-layer tiles, the tiles are
// tree-broadcast to every layer (Phase::Replication), and layer l executes the
// rounds [l*q/c, (l+1)*q/c) of the full q-round Cannon schedule with skew
// offset o = l*q/c, so node (layer, i, j) multiplies k = (i + j + o + round) % q.
// Every (i, k, j) product then runs exactly once across layers with q/c per
// node. Partial C tiles sum back to the base layer (Phase::Reduction) before
// the tiles convert to the common row shards.
MatmulDefinition build_two_point_five_d(BspPlan& plan, const MatmulConfig& config) {
    const auto N = config.padded_size, P = config.P, c = config.c;
    const auto q = grid_size(P, c);
    if (!N || N % P || plan.nodes() != P)
        throw std::invalid_argument("invalid 2.5D configuration");
    const auto block = N / q, height = N / P, rounds = q / c;
    const auto node_at = [q](std::size_t layer, std::size_t row, std::size_t col) {
        return layer * q * q + row * q + col;
    };
    MatmulDefinition definition{std::vector<MatrixView>(P), std::vector<MatrixView>(P),
                                std::vector<MatrixView>(P), {q, q, q}};
    std::vector<MatrixView> A(P), B(P), C(P), scratch(P);
    // c = 1 tiles fit the row-shard buffers and alias them through scratch,
    // matching Cannon's memory footprint; c > 1 keeps c-fold redundant tiles.
    const auto alias = block * block == height * N;
    for (std::size_t node = 0; node < P; ++node) {
        definition.input_a_rows[node] =
            plan.matrix_view(plan.allocate_buffer(node, height * N), height, N);
        definition.input_b_rows[node] =
            plan.matrix_view(plan.allocate_buffer(node, height * N), height, N);
        definition.output_c_rows[node] =
            plan.matrix_view(plan.allocate_buffer(node, height * N), height, N);
        if (alias) {
            A[node] = plan.matrix_view(definition.input_a_rows[node].buffer, block, block);
            B[node] = plan.matrix_view(definition.input_b_rows[node].buffer, block, block);
            C[node] = plan.matrix_view(definition.output_c_rows[node].buffer, block, block);
            if (P > 1)
                scratch[node] = plan.matrix_view(plan.allocate_buffer(node, height * block), height, block);
        } else {
            A[node] = plan.matrix_view(plan.allocate_buffer(node, block * block), block, block);
            B[node] = plan.matrix_view(plan.allocate_buffer(node, block * block), block, block);
            C[node] = plan.matrix_view(plan.allocate_buffer(node, block * block), block, block);
        }
    }
    add_rows_to_tiles(plan, config, definition.input_a_rows, A, scratch);
    add_rows_to_tiles(plan, config, definition.input_b_rows, B, scratch);
    std::vector<TreeGroup> groups;
    NodePayloads payloads(P);
    for (std::size_t i = 0; i < q; ++i)
        for (std::size_t j = 0; j < q; ++j) {
            TreeGroup group;
            for (std::size_t layer = 0; layer < c; ++layer)
                group.push_back(node_at(layer, i, j));
            groups.push_back(group);
            for (std::size_t layer = 0; layer < c; ++layer) {
                const auto node = node_at(layer, i, j);
                payloads[node] = {{A[node], {i * block, j * block}}, {B[node], {i * block, j * block}}};
            }
        }
    add_tree_broadcast(plan, groups, payloads);
    // All c layers run concurrently, so each layer's skew shares one superstep
    // and the layers' round r shares the round-r superstep.
    auto skew = plan.add_superstep(Phase::Skew);
    for (std::size_t layer = 0; layer < c; ++layer) {
        const auto offset = layer * rounds;
        for (std::size_t i = 0; i < q; ++i)
            for (std::size_t j = 0; j < q; ++j) {
                const auto node = node_at(layer, i, j);
                skew.add_zero(C[node]);
                if ((i + offset) % q)
                    skew.add_send(node_at(layer, i, (j + q - (i + offset) % q) % q),
                                  PayloadPart{A[node], {i * block, j * block}});
                if ((j + offset) % q)
                    skew.add_send(node_at(layer, (i + q - (j + offset) % q) % q, j),
                                  PayloadPart{B[node], {i * block, j * block}});
            }
    }
    for (std::size_t round = 0; round < rounds; ++round) {
        auto step = plan.add_superstep(Phase::Multiply);
        for (std::size_t layer = 0; layer < c; ++layer) {
            const auto offset = layer * rounds;
            for (std::size_t i = 0; i < q; ++i)
                for (std::size_t j = 0; j < q; ++j) {
                    const auto node = node_at(layer, i, j), k = (i + j + offset + round) % q;
                    const auto source_a = round ? node_at(layer, i, (j + 1) % q)
                                                : node_at(layer, i, (j + i + offset) % q);
                    const auto source_b = round ? node_at(layer, (i + 1) % q, j)
                                                : node_at(layer, (i + j + offset) % q, j);
                    if (source_a != node)
                        step.add_receive(source_a, A[node]);
                    if (source_b != node)
                        step.add_receive(source_b, B[node]);
                    step.add_gemm(A[node], B[node], C[node], {i, k, j});
                    if (round + 1 < rounds) {
                        step.add_send(node_at(layer, i, (j + q - 1) % q),
                                      PayloadPart{A[node], {i * block, k * block}});
                        step.add_send(node_at(layer, (i + q - 1) % q, j),
                                      PayloadPart{B[node], {k * block, j * block}});
                    }
                }
        }
    }
    groups.clear();
    for (std::size_t i = 0; i < q; ++i)
        for (std::size_t j = 0; j < q; ++j) {
            TreeGroup group;
            for (std::size_t layer = 0; layer < c; ++layer)
                group.push_back(node_at(layer, i, j));
            groups.push_back(group);
            for (std::size_t layer = 0; layer < c; ++layer) {
                const auto node = node_at(layer, i, j);
                payloads[node] = {{C[node], {i * block, j * block}}};
            }
        }
    add_tree_reduce_sum(plan, groups, payloads);
    add_tiles_to_rows(plan, config, C, definition.output_c_rows, scratch);
    return definition;
}
} // namespace dmm
