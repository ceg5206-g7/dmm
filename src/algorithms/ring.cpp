#include "dmm/algorithms.hpp"

namespace dmm {
// 1D ring: node i owns A row strip i, B row block i and C row strip i, so the
// common row shards are the working layout and no conversion is scheduled. B
// blocks travel around the ring; node i accumulates C_i = sum_k A_{i,k} B_k in
// P multiply supersteps with k = (i - round) % P.
MatmulDefinition build_ring(BspPlan& plan, const MatmulConfig& config) {
    if (config.c != 1)
        throw std::invalid_argument("ring requires c = 1");
    const auto N = config.padded_size, P = config.P;
    if (!P || !N || N % P || plan.nodes() != P)
        throw std::invalid_argument("invalid ring configuration");
    const auto height = N / P, words = detail::elements<float>(N) / P;
    MatmulDefinition definition{std::vector<MatrixView>(P), std::vector<MatrixView>(P),
                                std::vector<MatrixView>(P), {P, P, 1}};
    std::vector<MatrixView> A(P), B(P), C(P);
    for (std::size_t node = 0; node < P; ++node) {
        A[node] = plan.matrix_view(plan.allocate_buffer(node, words), height, N);
        B[node] = plan.matrix_view(plan.allocate_buffer(node, words), height, N);
        C[node] = plan.matrix_view(plan.allocate_buffer(node, words), height, N);
        definition.input_a_rows[node] = A[node];
        definition.input_b_rows[node] = B[node];
        definition.output_c_rows[node] = C[node];
    }
    for (std::size_t round = 0; round < P; ++round) {
        auto step = plan.add_superstep(Phase::Multiply);
        for (std::size_t node = 0; node < P; ++node) {
            const auto k = (node + P - round) % P;
            if (!round)
                step.add_zero(C[node]);
            if (round)
                step.add_receive((node + P - 1) % P, B[node]);
            step.add_gemm(A[node].submatrix(0, k * height, height, height), B[node], C[node],
                          {node, k, 0});
            if (round + 1 < P)
                step.add_send((node + 1) % P, PayloadPart{B[node], {k * height, 0}});
        }
    }
    return definition;
}
} // namespace dmm
