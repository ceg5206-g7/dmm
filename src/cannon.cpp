#include "dmm/cannon.hpp"
#include "dmm/matmul.hpp"
#include "dmm/sharding.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace dmm {

Traced Cannon::run(const Matrix<float>& a, const Matrix<float>& b, Executor& executor) const {
    return PreparedMatmul(a, b, P, c, build_cannon).run(executor);
}
Matrix<float> Cannon::matmul(const Matrix<float>& a, const Matrix<float>& b, Executor& executor) const {
    return run(a, b, executor).product;
}

void build_cannon(BspPlan& plan, MatmulLayout& layout) {
    const auto n = layout.n, P = layout.P, c = layout.c;
    const auto q = grid_size(P, c);
    if (c != 1)
        throw std::invalid_argument("Cannon requires c = 1");
    auto& left = layout.a;
    auto& right = layout.b;
    auto& result = layout.result;
    std::vector<std::size_t> scratch;
    const auto block = n / q, height = n / P, words = block * block;
    for (std::size_t node = 0; node < P; ++node) {
        left.push_back(plan.allocate(node, words));
        right.push_back(plan.allocate(node, words));
        result.push_back(plan.allocate(node, words));
        if (P > 1)
            scratch.push_back(plan.allocate(node, height * block));
    }
    auto v = [&](std::size_t id) { return BufferView{id, 0, block}; };
    if (P > 1) {
        rows_to_blocks(plan, n, 1, left, left, 0, scratch);
        rows_to_blocks(plan, n, 1, right, right, 1, scratch);
    }
    const auto skew = plan.step("skew");
    for (std::size_t node = 0; node < P; ++node)
        plan.zero(P > 1 ? 0 : skew, node, result[node]);
    const auto none = std::numeric_limits<std::size_t>::max();
    std::vector<std::size_t> pending_a(P, none), pending_b(P, none);
    auto at = [&](std::size_t i, std::size_t j) { return (i % q) * q + j % q; };
    for (std::size_t node = 0; node < P; ++node) {
        const auto i = node / q, j = node % q;
        if (i) {
            const auto dest = at(i, j + q - i);
            pending_a[dest] =
                plan.send(skew, node, dest, 0, v(left[node]), {i * block, j * block, block, block});
        }
        if (j) {
            const auto dest = at(i + q - j, j);
            pending_b[dest] =
                plan.send(skew, node, dest, 1, v(right[node]), {i * block, j * block, block, block});
        }
    }
    for (std::size_t round = 0; round < q; ++round) {
        const auto step = plan.step("multiply");
        for (std::size_t node = 0; node < P; ++node) {
            if (pending_a[node] != none)
                plan.receive(step, node, pending_a[node], v(left[node]));
            if (pending_b[node] != none)
                plan.receive(step, node, pending_b[node], v(right[node]));
            const auto i = node / q, j = node % q, k = (i + j + round) % q;
            plan.multiply_add(step, node, {i, k, j, block, block, block}, v(left[node]), v(right[node]),
                              v(result[node]));
        }
        if (round + 1 < q)
            for (std::size_t node = 0; node < P; ++node) {
                const auto i = node / q, j = node % q, k = (i + j + round) % q;
                const auto dest_a = at(i, j + q - 1), dest_b = at(i + q - 1, j);
                pending_a[dest_a] =
                    plan.send(step, node, dest_a, 0, v(left[node]), {i * block, k * block, block, block});
                pending_b[dest_b] =
                    plan.send(step, node, dest_b, 1, v(right[node]), {k * block, j * block, block, block});
            }
    }
    if (P > 1)
        blocks_to_rows(plan, n, 1, result, result, 2, scratch);
    layout.coverage = {q, q, q};
}

Error validate(const Matrix<float>& result, const Matrix<double>& reference, double atol, double rtol) {
    if (result.size != reference.size || result.values.size() != detail::elements<float>(result.size) ||
        reference.values.size() != detail::elements<double>(reference.size) || atol < 0 || rtol < 0 ||
        !std::isfinite(atol) || !std::isfinite(rtol))
        throw std::invalid_argument("invalid reference shape or tolerances");
    double max_error = 0, squared_error = 0, squared_reference = 0;
    for (std::size_t i = 0; i < reference.values.size(); ++i) {
        const double value = result.values[i], expected = reference.values[i];
        const auto error = std::abs(value - expected);
        if (!std::isfinite(value) || !std::isfinite(expected) || error > atol + rtol * std::abs(expected))
            throw std::runtime_error("incorrect matrix result at element " + std::to_string(i));
        max_error = std::max(max_error, error);
        squared_error += error * error;
        squared_reference += expected * expected;
    }
    return {max_error,
            squared_reference == 0 ? std::sqrt(squared_error) : std::sqrt(squared_error / squared_reference)};
}

} // namespace dmm
