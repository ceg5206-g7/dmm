#include "dmm/matmul.hpp"
#include "dmm/sharding.hpp"
#include <algorithm>
#include <cmath>

namespace dmm {

PreparedMatmul::PreparedMatmul(Matrix<float> a, Matrix<float> b, std::size_t P, std::size_t c,
                               PlanBuilder build)
    : a_(std::move(a)), b_(std::move(b)), layout_{padded_size(a_.size, P), P, c, {}, {}, {}}, plan_(P) {
    if (!c || !build || a_.size != b_.size || a_.values.size() != detail::elements<float>(a_.size) ||
        b_.values.size() != detail::elements<float>(b_.size))
        throw std::invalid_argument("invalid matrix inputs, replication or builder");
    for (const auto* input : {&a_, &b_})
        for (auto value : input->values)
            if (!std::isfinite(value))
                throw std::invalid_argument("inputs must be finite");
    build(plan_, layout_);
    const auto words = layout_.n * (layout_.n / P);
    for (const auto* ids : {&layout_.a, &layout_.b, &layout_.result}) {
        if (ids->size() != P)
            throw std::invalid_argument("builder must provide one row buffer per node");
        for (std::size_t node = 0; node < P; ++node)
            if (!plan_.owns(node, (*ids)[node]) || plan_.buffer((*ids)[node]).size() < words)
                throw std::invalid_argument("builder row buffers have wrong owner or capacity");
    }
    // Inputs are independently restored; aliasing them or the output would destroy input data.
    for (std::size_t node = 0; node < P; ++node)
        if (layout_.a[node] == layout_.b[node] || layout_.a[node] == layout_.result[node] ||
            layout_.b[node] == layout_.result[node])
            throw std::invalid_argument("row inputs and output must be distinct");
    plan_.seal(a_.size, layout_.n, layout_.coverage);
}
RunTimings PreparedMatmul::run_into(std::span<float> output, Executor& executor) {
    std::lock_guard lock(mutex_);
    return execute_into(output, executor);
}
RunTimings PreparedMatmul::execute_into(std::span<float> output, Executor& executor) {
    if (output.size() != a_.values.size())
        throw std::invalid_argument("prepared output size mismatch");
    const auto n = layout_.n, height = n / layout_.P;
    for (std::size_t node = 0; node < layout_.P; ++node)
        for (auto pair : {std::pair{&a_, layout_.a[node]}, std::pair{&b_, layout_.b[node]}}) {
            auto data = plan_.buffer(pair.second);
            std::fill(data.begin(), data.end(), 0.f);
            for (std::size_t row = 0; row < height && node * height + row < a_.size; ++row)
                std::copy_n(pair.first->values.data() + (node * height + row) * a_.size, a_.size,
                            data.data() + row * n);
        }
    auto timing = plan_.execute(executor);
    for (std::size_t node = 0; node < layout_.P; ++node) {
        auto data = plan_.buffer(layout_.result[node]);
        for (std::size_t row = 0; row < height && node * height + row < a_.size; ++row)
            std::copy_n(data.data() + row * n, a_.size, output.data() + (node * height + row) * a_.size);
    }
    return timing;
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
