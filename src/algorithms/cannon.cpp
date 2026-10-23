#include "dmm/algorithms.hpp"
#include "dmm/redistribution.hpp"

namespace dmm {
MatmulDefinition build_cannon(BspPlan& plan, const MatmulConfig& config) {
    if (config.c != 1)
        throw std::invalid_argument("Cannon requires c = 1");
    const auto N = config.padded_size, P = config.P, q = grid_size(P);
    if (!N || N % P || plan.nodes() != P)
        throw std::invalid_argument("invalid Cannon configuration");
    const auto block = N / q, height = N / P;
    std::vector<MatrixView> A(P), B(P), C(P), scratch(P);
    MatmulDefinition definition{
        std::vector<MatrixView>(P), std::vector<MatrixView>(P), std::vector<MatrixView>(P), {q, q, q}};
    auto allocate_matrix = [&](std::size_t node, std::size_t rows, std::size_t cols) {
        return plan.matrix_view(plan.allocate_buffer(node, rows * cols), rows, cols);
    };
    for (std::size_t node = 0; node < P; ++node) {
        A[node] = allocate_matrix(node, block, block);
        B[node] = allocate_matrix(node, block, block);
        C[node] = allocate_matrix(node, block, block);
        definition.input_a_rows[node] = plan.matrix_view(A[node].buffer, height, N);
        definition.input_b_rows[node] = plan.matrix_view(B[node].buffer, height, N);
        definition.output_c_rows[node] = plan.matrix_view(C[node].buffer, height, N);
        if (P > 1)
            scratch[node] = allocate_matrix(node, height, block);
    }
    if (P > 1) {
        add_rows_to_tiles(plan, config, definition.input_a_rows, A, scratch);
        add_rows_to_tiles(plan, config, definition.input_b_rows, B, scratch);
    }
    auto node_at = [q](std::size_t row, std::size_t col) { return (row % q) * q + col % q; };
    auto skew = plan.add_superstep(Phase::Skew);
    for (std::size_t node = 0; node < P; ++node) {
        const auto i = node / q, j = node % q;
        skew.add_zero(C[node]);
        if (i)
            skew.add_send(node_at(i, j + q - i), PayloadPart{A[node], {i * block, j * block}});
        if (j)
            skew.add_send(node_at(i + q - j, j), PayloadPart{B[node], {i * block, j * block}});
    }
    for (std::size_t round = 0; round < q; ++round) {
        auto step = plan.add_superstep(Phase::Multiply);
        for (std::size_t node = 0; node < P; ++node) {
            const auto i = node / q, j = node % q, k = (i + j + round) % q;
            const auto source_a = node_at(i, j + (round == 0 ? i : 1));
            const auto source_b = node_at(i + (round == 0 ? j : 1), j);
            if (source_a != node)
                step.add_receive(source_a, A[node]);
            if (source_b != node)
                step.add_receive(source_b, B[node]);
            step.add_gemm(A[node], B[node], C[node], {i, k, j});
            if (round + 1 < q) {
                step.add_send(node_at(i, j + q - 1), PayloadPart{A[node], {i * block, k * block}});
                step.add_send(node_at(i + q - 1, j), PayloadPart{B[node], {k * block, j * block}});
            }
        }
    }
    if (P > 1)
        add_tiles_to_rows(plan, config, C, definition.output_c_rows, scratch);
    return definition;
}
} // namespace dmm
