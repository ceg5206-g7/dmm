#include "dmm/cannon.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace dmm {
namespace {

struct Task {
    std::size_t row, col;
    std::array<std::pair<MatrixBlockView<float>, MatrixBlockView<float>>, GRID> pairs;
};

std::vector<Task> build_tasks(const Matrix<float>& a, const Matrix<float>& b) {
    const auto n = a.size, block = n / GRID;
    std::vector<Task> tasks;
    tasks.reserve(GRID * GRID);
    for (std::size_t i = 0; i < GRID; ++i)
        for (std::size_t j = 0; j < GRID; ++j) {
            Task task{i, j, {}};
            for (std::size_t stage = 0; stage < GRID; ++stage) {
                // shortcut: shifts are represented by views; implement message exchange when adding communication simulation.
                const auto k = (i + j + stage) % GRID;
                task.pairs[stage] = {a.block(i * block, k * block, block),
                                     b.block(k * block, j * block, block)};
            }
            tasks.push_back(task);
        }
    return tasks;
}

std::vector<float> compute_tile(const Task& task) {
    const auto block = task.pairs[0].first.size;
    std::vector<float> tile(block * block, 0.0f);
    for (const auto& [a, b] : task.pairs)
        for (std::size_t i = 0; i < block; ++i) {
            auto* output_row = tile.data() + i * block;
            for (std::size_t k = 0; k < block; ++k) {
                const auto value = a(i, k);
                for (std::size_t j = 0; j < block; ++j)
                    output_row[j] += value * b(k, j);
            }
        }
    return tile;
}

} // namespace

Matrix<float> Cannon::matmul(const Matrix<float>& a, const Matrix<float>& b, Executor& executor) const {
    if (a.size != b.size || a.values.size() != detail::elements<float>(a.size) ||
        b.values.size() != detail::elements<float>(b.size))
        throw std::invalid_argument("inputs must be matching square matrices");
    if (a.size % GRID != 0)
        throw std::invalid_argument("Cannon matrix size must be divisible by 4");
    const auto tasks = build_tasks(a, b);
    const auto n = a.size, block = n / GRID;
    std::vector<std::vector<float>> tiles(tasks.size());
    executor.for_each(tasks.size(), [&](std::size_t index) {
        tiles[index] = compute_tile(tasks[index]);
    });
    Matrix<float> output(n);
    for (std::size_t index = 0; index < tasks.size(); ++index)
        for (std::size_t i = 0; i < block; ++i)
            std::copy_n(tiles[index].data() + i * block, block,
                        output.values.data() + (tasks[index].row * block + i) * n + tasks[index].col * block);
    return output;
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
