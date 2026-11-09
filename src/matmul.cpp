#include "dmm/matmul.hpp"
#include "dmm/redistribution.hpp"
#include <cmath>

namespace dmm {
PreparedMatmul::PreparedMatmul(Matrix<float> a, Matrix<float> b, std::size_t P, std::size_t c,
                               PlanBuilder build)
    : a_(std::move(a)), b_(std::move(b)), config_{padded_size(a_.size, P), P, c}, plan_(P) {
    if (!c || !build || a_.size != b_.size || a_.values.size() != detail::elements<float>(a_.size) ||
        b_.values.size() != detail::elements<float>(b_.size))
        throw std::invalid_argument("invalid matrix inputs, replication or builder");
    for (const auto* input : {&a_, &b_})
        for (auto value : input->values)
            if (!std::isfinite(value))
                throw std::invalid_argument("inputs must be finite");
    definition_ = build(plan_, config_);
    const auto height = config_.padded_size / P;
    for (const auto* views :
         {&definition_.input_a_rows, &definition_.input_b_rows, &definition_.output_c_rows}) {
        if (views->size() != P)
            throw std::invalid_argument("builder must bind one row view per node");
        for (std::size_t node = 0; node < P; ++node) {
            const auto view = (*views)[node];
            if (plan_.owner(view) != node || view.rows != height || view.cols != config_.padded_size)
                throw std::invalid_argument("builder row views have wrong owner or shape");
        }
    }
    for (std::size_t node = 0; node < P; ++node) {
        const auto a_id = definition_.input_a_rows[node].buffer, b_id = definition_.input_b_rows[node].buffer,
                   c_id = definition_.output_c_rows[node].buffer;
        if (a_id == b_id || a_id == c_id || b_id == c_id)
            throw std::invalid_argument("input/output row buffers must be distinct");
        plan_.mark_host_input(a_id);
        plan_.mark_host_input(b_id);
        plan_.mark_host_output(c_id);
    }
    plan_.finalize(a_.size, config_.padded_size, definition_.product_grid, true);
}
RunTimings PreparedMatmul::execute_into(std::span<float> output, Executor& executor) {
    if (output.size() != a_.values.size())
        throw std::invalid_argument("prepared output size mismatch");
    const auto height = config_.padded_size / config_.P;
    for (std::size_t node = 0; node < config_.P; ++node)
        for (auto pair : {std::pair{&a_, definition_.input_a_rows[node]},
                          std::pair{&b_, definition_.input_b_rows[node]}}) {
            auto data = plan_.buffer(pair.second.buffer).subspan(pair.second.offset);
            for (std::size_t row = 0; row < height; ++row) {
                std::fill_n(data.data() + row * pair.second.stride, pair.second.cols, 0.f);
                if (node * height + row < a_.size)
                    std::copy_n(pair.first->values.data() + (node * height + row) * a_.size, a_.size,
                                data.data() + row * pair.second.stride);
            }
        }
    auto timing = plan_.execute(executor);
    for (std::size_t node = 0; node < config_.P; ++node) {
        const auto view = definition_.output_c_rows[node];
        auto data = plan_.buffer(view.buffer).subspan(view.offset);
        for (std::size_t row = 0; row < height && node * height + row < a_.size; ++row)
            std::copy_n(data.data() + row * view.stride, a_.size,
                        output.data() + (node * height + row) * a_.size);
    }
    return timing;
}
RunTimings PreparedMatmul::run_into(std::span<float> output, Executor& executor) {
    std::lock_guard lock(mutex_);
    return execute_into(output, executor);
}
Trace PreparedMatmul::trace() const {
    std::lock_guard lock(mutex_);
    return plan_.trace();
}
Traced PreparedMatmul::run(Executor& executor) {
    std::lock_guard lock(mutex_);
    Matrix<float> product(size());
    execute_into(product.values, executor);
    return {std::move(product), plan_.trace()};
}
} // namespace dmm
