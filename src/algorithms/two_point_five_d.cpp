#include "dmm/algorithms.hpp"
#include "dmm/collectives.hpp"
#include "dmm/redistribution.hpp"

namespace dmm {
namespace {
// All c layers run concurrently, so each layer's skew shares one superstep
// and the layers' round r shares the round-r superstep. Layer l starts at skew
// offset o = l*q/c and runs q/c rounds, so (i, k, j) runs once across layers.
void add_layer_cannon(BspPlan& plan, std::size_t q, std::size_t c, std::size_t block,
                      std::span<const MatrixView> A, std::span<const MatrixView> B,
                      std::span<const MatrixView> C) {
    const auto rounds = q / c;
    const auto node_at = [q](std::size_t layer, std::size_t row, std::size_t col) {
        return layer * q * q + row * q + col;
    };
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
}
} // namespace

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
    const auto block = N / q, height = N / P;
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
    const ConversionMatrix inputs[]{{definition.input_a_rows, A}, {definition.input_b_rows, B}};
    add_rows_to_tiles(plan, config, inputs, scratch);
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
    add_layer_cannon(plan, q, c, block, A, B, C);
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
// Direct 2.5D (an ablation of the proposal's base-layer route): every row
// shard sends its pieces straight to the tiles of all c layers, which removes
// the base-layer gather and the broadcast tree. Each node sends and receives
// 2c*n^2/P words in one exchange. After the shared layer-Cannon body, each
// layer's partial C tiles go straight to the row-shard owners, which sum the c
// contributions per block (a reduce-scatter of c*n^2/P words per node). There
// are no separate conversion phases: the exchanges are labelled replication
// and reduction, so the native and end-to-end counts coincide, as for the ring.
// c = 1 has nothing to replicate and uses the base-layer route, i.e. Cannon.
MatmulDefinition build_two_point_five_d_direct(BspPlan& plan, const MatmulConfig& config) {
    if (config.c == 1)
        return build_two_point_five_d(plan, config);
    const auto N = config.padded_size, P = config.P, c = config.c;
    const auto q = grid_size(P, c);
    if (!N || N % P || plan.nodes() != P)
        throw std::invalid_argument("invalid 2.5D configuration");
    // Tile row i holds rows [i*block, (i+1)*block), i.e. row shards
    // [i*shards, (i+1)*shards) with shards = block / height = c*q.
    const auto block = N / q, height = N / P, shards = block / height;
    const auto node_at = [q](std::size_t layer, std::size_t row, std::size_t col) {
        return layer * q * q + row * q + col;
    };
    MatmulDefinition definition{std::vector<MatrixView>(P), std::vector<MatrixView>(P),
                                std::vector<MatrixView>(P), {q, q, q}};
    std::vector<MatrixView> A(P), B(P), C(P);
    auto allocate_matrix = [&](std::size_t node, std::size_t rows, std::size_t cols) {
        return plan.matrix_view(plan.allocate_buffer(node, rows * cols), rows, cols);
    };
    for (std::size_t node = 0; node < P; ++node) {
        definition.input_a_rows[node] = allocate_matrix(node, height, N);
        definition.input_b_rows[node] = allocate_matrix(node, height, N);
        definition.output_c_rows[node] = allocate_matrix(node, height, N);
        A[node] = allocate_matrix(node, block, block);
        B[node] = allocate_matrix(node, block, block);
        C[node] = allocate_matrix(node, block, block);
    }
    // The (shard r, column block j) piece of a row shard and its place in tile t.
    const auto row_piece = [&](const std::vector<MatrixView>& rows, std::size_t r, std::size_t j) {
        return rows[r].submatrix(0, j * block, height, block);
    };
    const auto tile_piece = [&](const std::vector<MatrixView>& tiles, std::size_t t, std::size_t r) {
        return tiles[t].submatrix(r * height - r / shards * block, 0, height, block);
    };
    // Local pieces move during the send step; a node copies before it sends.
    auto replicate = plan.add_superstep(Phase::Replication);
    for (bool sending : {false, true})
        for (std::size_t r = 0; r < P; ++r)
            for (std::size_t layer = 0; layer < c; ++layer)
                for (std::size_t j = 0; j < q; ++j) {
                    const auto t = node_at(layer, r / shards, j);
                    if (t == r && !sending) {
                        replicate.add_copy(row_piece(definition.input_a_rows, r, j), tile_piece(A, t, r));
                        replicate.add_copy(row_piece(definition.input_b_rows, r, j), tile_piece(B, t, r));
                    } else if (t != r && sending)
                        replicate.add_send(t, {PayloadPart{row_piece(definition.input_a_rows, r, j),
                                                           {r * height, j * block}},
                                               PayloadPart{row_piece(definition.input_b_rows, r, j),
                                                           {r * height, j * block}}});
                }
    auto deliver = plan.add_superstep(Phase::Replication);
    for (std::size_t r = 0; r < P; ++r)
        for (std::size_t layer = 0; layer < c; ++layer)
            for (std::size_t j = 0; j < q; ++j)
                if (const auto t = node_at(layer, r / shards, j); t != r)
                    deliver.add_receive(r, {tile_piece(A, t, r), tile_piece(B, t, r)});
    add_layer_cannon(plan, q, c, block, A, B, C);
    // Row shard r receives block j of its rows from tile (layer, r / shards, j)
    // on every layer. Its own piece, if any, is copied in the send step; the
    // first remote piece then overwrites or adds, in layer order.
    auto scatter = plan.add_superstep(Phase::Reduction);
    for (bool sending : {false, true})
        for (std::size_t t = 0; t < P; ++t) {
            const auto i = t % (q * q) / q, j = t % q;
            for (auto r = i * shards; r < (i + 1) * shards; ++r) {
                if (t == r && !sending)
                    scatter.add_copy(tile_piece(C, t, r), row_piece(definition.output_c_rows, r, j));
                else if (t != r && sending)
                    scatter.add_send(r, PayloadPart{tile_piece(C, t, r), {r * height, j * block}});
            }
        }
    auto sum = plan.add_superstep(Phase::Reduction);
    for (std::size_t r = 0; r < P; ++r)
        for (std::size_t j = 0; j < q; ++j) {
            bool written = false;
            for (std::size_t layer = 0; layer < c; ++layer)
                written = written || node_at(layer, r / shards, j) == r;
            for (std::size_t layer = 0; layer < c; ++layer) {
                const auto t = node_at(layer, r / shards, j);
                if (t == r)
                    continue;
                const auto target = row_piece(definition.output_c_rows, r, j);
                written ? sum.add_receive_sum(t, target) : sum.add_receive(t, target);
                written = true;
            }
        }
    return definition;
}
} // namespace dmm
