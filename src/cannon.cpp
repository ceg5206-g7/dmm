#include "dmm/cannon.hpp"
#include "dmm/prepared.hpp"
#include "dmm/sharding.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace dmm {

Traced Cannon::run(const Matrix<float>& a, const Matrix<float>& b, Executor& executor) const {
    return PlannedAlgorithm<PreparedCannon>{P, c}.run(a, b, executor);
}

Traced Cannon::run_native(const Matrix<float>& a, const Matrix<float>& b, Executor& executor) const {
    return run_once(PreparedCannon(a, b, P, c, true), executor);
}

Matrix<float> Cannon::matmul(const Matrix<float>& a, const Matrix<float>& b, Executor& executor) const {
    return run(a, b, executor).product;
}

PreparedCannon::PreparedCannon(Matrix<float> a, Matrix<float> b, std::size_t P, std::size_t c)
    : PreparedCannon(std::move(a), std::move(b), P, c, false) {}

PreparedCannon::PreparedCannon(Matrix<float> a, Matrix<float> b, std::size_t P, std::size_t c, bool native_layout)
    : a_(std::move(a)), b_(std::move(b)), P_(P),
      n_(native_layout ? a_.size : padded_size(a_.size, P)), q_(grid_size(P, c)),
      native_layout_(native_layout), plan_(P) {
    if (c != 1 || a_.size != b_.size || a_.values.size() != detail::elements<float>(a_.size) ||
        b_.values.size() != detail::elements<float>(b_.size)) throw std::invalid_argument("invalid prepared Cannon inputs/grid");
    if (native_layout_ && n_ % q_ != 0)
        throw std::invalid_argument("native Cannon matrix size must be divisible by " + std::to_string(q_));
    for (const auto* input : {&a_, &b_})
        for (auto value : input->values) if (!std::isfinite(value)) throw std::invalid_argument("inputs must be finite");
    const auto block = n_ / q_, height = n_ / P_, words = block * block;
    for (std::size_t node = 0; node < P_; ++node) {
        left_.push_back(plan_.allocate(node, words)); right_.push_back(plan_.allocate(node, words));
        result_.push_back(plan_.allocate(node, words));
        if (!native_layout_ && P_ > 1) scratch_.push_back(plan_.allocate(node, height * block));
    }
    auto v = [&](std::size_t id) { return BufferView{id, 0, block}; };
    if (!native_layout_ && P_ > 1) {
        rows_to_blocks(plan_,n_,1,left_,left_,0,scratch_);
        rows_to_blocks(plan_,n_,1,right_,right_,1,scratch_);
    }
    const auto skew = plan_.step("skew");
    for (std::size_t node = 0; node < P_; ++node) plan_.zero(!native_layout_ && P_ > 1 ? 0 : skew, node, result_[node]);
    const auto none = std::numeric_limits<std::size_t>::max();
    std::vector<std::size_t> pending_a(P_, none), pending_b(P_, none);
    auto at = [&](std::size_t i, std::size_t j) { return (i % q_) * q_ + j % q_; };
    for (std::size_t node = 0; node < P_; ++node) {
        const auto i = node / q_, j = node % q_;
        if (i) { const auto dest = at(i, j + q_ - i);
            pending_a[dest] = plan_.send(skew, node, dest, 0, v(left_[node]), {i * block, j * block, block, block}); }
        if (j) { const auto dest = at(i + q_ - j, j);
            pending_b[dest] = plan_.send(skew, node, dest, 1, v(right_[node]), {i * block, j * block, block, block}); }
    }
    for (std::size_t round = 0; round < q_; ++round) {
        const auto step = plan_.step("multiply");
        for (std::size_t node = 0; node < P_; ++node) {
            if (pending_a[node] != none) plan_.receive(step, node, pending_a[node], v(left_[node]));
            if (pending_b[node] != none) plan_.receive(step, node, pending_b[node], v(right_[node]));
            const auto i = node / q_, j = node % q_, k = (i + j + round) % q_;
            plan_.multiply_add(step, node, {i, k, j, block, block, block}, v(left_[node]), v(right_[node]), v(result_[node]));
        }
        if (round + 1 < q_) for (std::size_t node = 0; node < P_; ++node) {
            const auto i = node / q_, j = node % q_, k = (i + j + round) % q_;
            const auto dest_a = at(i, j + q_ - 1), dest_b = at(i + q_ - 1, j);
            pending_a[dest_a] = plan_.send(step, node, dest_a, 0, v(left_[node]), {i * block, k * block, block, block});
            pending_b[dest_b] = plan_.send(step, node, dest_b, 1, v(right_[node]), {k * block, j * block, block, block});
        }
    }
    if (!native_layout_ && P_ > 1) blocks_to_rows(plan_,n_,1,result_,result_,2,scratch_);
    plan_.seal(a_.size, n_, {q_, q_, q_});
}
RunTimings PreparedCannon::run_into(std::span<float> output, Executor& executor) {
    std::lock_guard lock(mutex_);
    if (output.size() != a_.values.size()) throw std::invalid_argument("prepared output size mismatch");
    const auto height = n_ / P_, block = n_ / q_;
    for (std::size_t node = 0; node < P_; ++node)
        for (auto pair : {std::pair{&a_, left_[node]}, std::pair{&b_, right_[node]}}) {
            auto data = plan_.buffer(pair.second);
            if (native_layout_) {
                for (std::size_t row = 0; row < block; ++row)
                    std::copy_n(pair.first->values.data() + (node / q_ * block + row) * n_ + node % q_ * block,
                                block, data.data() + row * block);
            } else {
                std::fill(data.begin(), data.end(), 0.f);
                for (std::size_t row = 0; row < height && node * height + row < a_.size; ++row)
                    std::copy_n(pair.first->values.data() + (node * height + row) * a_.size, a_.size,
                                data.data() + row * n_);
            }
        }
    auto timing = plan_.execute(executor);
    for (std::size_t node = 0; node < P_; ++node) {
        auto data = plan_.buffer(result_[node]);
        if (native_layout_) {
            for (std::size_t row = 0; row < block; ++row)
                std::copy_n(data.data() + row * block, block,
                            output.data() + (node / q_ * block + row) * n_ + node % q_ * block);
        } else {
            for (std::size_t row = 0; row < height && node * height + row < a_.size; ++row)
                std::copy_n(data.data() + row * n_, a_.size, output.data() + (node * height + row) * a_.size);
        }
    }
    return timing;
}
Trace PreparedCannon::trace() const { std::lock_guard lock(mutex_); return plan_.trace(); }

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
