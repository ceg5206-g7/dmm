#include "dmm/cannon.hpp"
#include "dmm/gemm.hpp"
#include "dmm/sharding.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace dmm {
namespace {
enum Tag : std::size_t { A_BLOCK, B_BLOCK, C_BLOCK };
std::size_t node_at(std::size_t row, std::size_t col, std::size_t q) { return row % q * q + col % q; }

void check_inputs(const Matrix<float>& a, const Matrix<float>& b) {
    if (a.size != b.size || a.values.size() != detail::elements<float>(a.size) ||
        b.values.size() != detail::elements<float>(b.size))
        throw std::invalid_argument("inputs must be matching square matrices");
}

NodeBuffers multiply_blocks(Bsp& machine, Executor& executor, NodeBuffers a,
                             NodeBuffers b, std::size_t n, std::size_t q) {
    const auto block = n / q, words = block * block;
    auto c = machine.buffers(words);
    machine.superstep("skew", executor, [&](Node& node) {
        const auto id = node.id(), i = id / q, j = id % q;
        if (i != 0) node.send(node_at(i, j + q - i, q), A_BLOCK, a[id],
                              ElementRegion{i * block, j * block, block, block});
        if (j != 0) node.send(node_at(i + q - j, j, q), B_BLOCK, b[id],
                              ElementRegion{i * block, j * block, block, block});
    });
    for (std::size_t round = 0; round < q; ++round)
        machine.superstep("multiply", executor, [&](Node& node) {
            const auto id = node.id(), i = id / q, j = id % q, k = (i + j + round) % q;
            if (round == 0) {
                if (i != 0) a[id] = node.receive(node_at(i, j + i, q), A_BLOCK, words);
                if (j != 0) b[id] = node.receive(node_at(i + j, j, q), B_BLOCK, words);
            } else {
                a[id] = node.receive(node_at(i, j + 1, q), A_BLOCK, words);
                b[id] = node.receive(node_at(i + 1, j, q), B_BLOCK, words);
            }
            node.multiply_add({i, k, j, block, block, block},
                              a[id], block, b[id], block, c[id], block);
            if (round + 1 < q) {
                node.send(node_at(i, j + q - 1, q), A_BLOCK, a[id],
                          ElementRegion{i * block, k * block, block, block});
                node.send(node_at(i + q - 1, j, q), B_BLOCK, b[id],
                          ElementRegion{k * block, j * block, block, block});
            }
        });
    return c;
}

NodeBuffers native_blocks(Bsp& machine, const Matrix<float>& input, std::size_t q) {
    const auto block = input.size / q;
    auto values = machine.buffers(block * block);
    for (std::size_t id = 0; id < values.size(); ++id)
        for (std::size_t i = 0; i < block; ++i)
            std::copy_n(input.values.data() + (id / q * block + i) * input.size + id % q * block,
                        block, values[id].data() + i * block);
    return values;
}
} // namespace

Traced Cannon::run(const Matrix<float>& a, const Matrix<float>& b, Executor& executor) const {
    check_inputs(a, b);
    const auto q = grid_size(P, c);
    if (c != 1) throw std::invalid_argument("Cannon requires c = 1");
    Bsp machine(P);
    auto rows_a = make_row_shards(machine, a);
    auto left = rows_to_blocks(machine, executor, rows_a, 1, A_BLOCK);
    rows_a.values.clear();
    auto rows_b = make_row_shards(machine, b);
    auto right = rows_to_blocks(machine, executor, rows_b, 1, B_BLOCK);
    rows_b.values.clear();
    const auto n = left.size;
    auto tiles = multiply_blocks(machine, executor, std::move(left.values), std::move(right.values), n, q);
    BlockShards product{a.size, n, 1, std::move(tiles)};
    auto rows_c = blocks_to_rows(machine, executor, product, C_BLOCK);
    product.values.clear();
    auto trace = machine.finish();
    trace.original_size = a.size; trace.padded_size = n; trace.coverage = {q, q, q};
    return {gather_row_shards(rows_c), std::move(trace)};
}

Traced Cannon::run_native(const Matrix<float>& a, const Matrix<float>& b, Executor& executor) const {
    check_inputs(a, b);
    const auto q = grid_size(P, c);
    if (c != 1) throw std::invalid_argument("Cannon requires c = 1");
    if (a.size % q != 0)
        throw std::invalid_argument("native Cannon matrix size must be divisible by " + std::to_string(q));
    Bsp machine(P);
    auto left = native_blocks(machine, a, q), right = native_blocks(machine, b, q);
    auto tiles = multiply_blocks(machine, executor, std::move(left), std::move(right), a.size, q);
    Matrix<float> output(a.size);
    const auto block = a.size / q;
    for (std::size_t id = 0; id < P; ++id)
        for (std::size_t i = 0; i < block; ++i)
            std::copy_n(tiles[id].data() + i * block, block,
                        output.values.data() + (id / q * block + i) * a.size + id % q * block);
    auto trace = machine.finish();
    trace.original_size = a.size; trace.padded_size = a.size; trace.coverage = {q, q, q};
    return {std::move(output), std::move(trace)};
}

Matrix<float> Cannon::matmul(const Matrix<float>& a, const Matrix<float>& b, Executor& executor) const {
    return run(a, b, executor).product;
}

Error validate(const Matrix<float>& result, const Matrix<double>& reference,
               double atol, double rtol) {
    if (result.size != reference.size ||
        result.values.size() != detail::elements<float>(result.size) ||
        reference.values.size() != detail::elements<double>(reference.size) || atol < 0 || rtol < 0 ||
        !std::isfinite(atol) || !std::isfinite(rtol))
        throw std::invalid_argument("invalid reference shape or tolerances");
    double max_error = 0, squared_error = 0, squared_reference = 0;
    for (std::size_t i = 0; i < reference.values.size(); ++i) {
        const double value = result.values[i], expected = reference.values[i];
        const auto error = std::abs(value - expected);
        if (!std::isfinite(value) || !std::isfinite(expected) ||
            error > atol + rtol * std::abs(expected))
            throw std::runtime_error("incorrect matrix result at element " + std::to_string(i));
        max_error = std::max(max_error, error);
        squared_error += error * error;
        squared_reference += expected * expected;
    }
    return {max_error, squared_reference == 0 ? std::sqrt(squared_error)
                                            : std::sqrt(squared_error / squared_reference)};
}

} // namespace dmm
