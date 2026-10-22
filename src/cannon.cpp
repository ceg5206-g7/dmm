#include "dmm/cannon.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace dmm {
namespace {

enum Tag : std::size_t { A_BLOCK, B_BLOCK };

// Node-private blocks; node code reads and writes only its own entry.
struct Local {
    std::vector<float> a, b, c;
};

std::size_t node_at(std::size_t row, std::size_t col, std::size_t grid) { return row % grid * grid + col % grid; }

std::vector<float> load_block(const Matrix<float>& matrix, std::size_t row, std::size_t col, std::size_t block) {
    const auto view = matrix.block(row * block, col * block, block);
    std::vector<float> values(block * block);
    for (std::size_t i = 0; i < block; ++i) std::copy_n(&view(i, 0), block, values.data() + i * block);
    return values;
}

void multiply_add(Local& local, std::size_t block) {
    for (std::size_t i = 0; i < block; ++i) {
        auto* output_row = local.c.data() + i * block;
        for (std::size_t k = 0; k < block; ++k) {
            const auto value = local.a[i * block + k];
            for (std::size_t j = 0; j < block; ++j)
                output_row[j] += value * local.b[k * block + j];
        }
    }
}

} // namespace

Traced Cannon::run(const Matrix<float>& a, const Matrix<float>& b, Executor& executor) const {
    if (a.size != b.size || a.values.size() != detail::elements<float>(a.size) ||
        b.values.size() != detail::elements<float>(b.size))
        throw std::invalid_argument("inputs must be matching square matrices");
    const auto grid = grid_size(P, c);
    if (c != 1) throw std::invalid_argument("Cannon requires c = 1");
    if (a.size % grid != 0)
        throw std::invalid_argument("Cannon matrix size must be divisible by " + std::to_string(grid));
    const auto n = a.size, block = n / grid, words = block * block;
    std::vector<Local> locals(grid * grid);
    Bsp machine(locals.size());

    // Node (i, j) starts with A_ij and B_ij, then skews A left by i and B up by j.
    machine.superstep("skew", executor, [&](Node& node) {
        const auto i = node.id() / grid, j = node.id() % grid;
        auto& local = locals[node.id()];
        local.a = load_block(a, i, j, block);
        local.b = load_block(b, i, j, block);
        local.c.assign(words, 0.0f);
        if (i != 0) node.send(node_at(i, j + grid - i, grid), A_BLOCK, local.a);
        if (j != 0) node.send(node_at(i + grid - j, j, grid), B_BLOCK, local.b);
    });
    // Round s multiplies A_{i,k} B_{k,j} with k = (i + j + s) % grid, then shifts A left and B up.
    for (std::size_t round = 0; round < grid; ++round)
        machine.superstep("multiply", executor, [&](Node& node) {
            const auto i = node.id() / grid, j = node.id() % grid;
            auto& local = locals[node.id()];
            if (round == 0) {
                if (i != 0) local.a = node.receive(node_at(i, j + i, grid), A_BLOCK, words);
                if (j != 0) local.b = node.receive(node_at(i + j, j, grid), B_BLOCK, words);
            } else {
                local.a = node.receive(node_at(i, j + 1, grid), A_BLOCK, words);
                local.b = node.receive(node_at(i + 1, j, grid), B_BLOCK, words);
            }
            multiply_add(local, block);
            node.compute(2 * words * block);
            if (round + 1 < grid) {
                node.send(node_at(i, j + grid - 1, grid), A_BLOCK, local.a);
                node.send(node_at(i + grid - 1, j, grid), B_BLOCK, local.b);
            }
        });
    auto trace = machine.finish();

    Matrix<float> output(n);
    for (std::size_t id = 0; id < locals.size(); ++id) {
        const auto row = id / grid, col = id % grid;
        for (std::size_t i = 0; i < block; ++i)
            std::copy_n(locals[id].c.data() + i * block, block,
                        output.values.data() + (row * block + i) * n + col * block);
    }
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
