#include "dmm/validation.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace dmm {
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

void validate_coverage(const Trace& trace) {
    const auto n = trace.padded_size;
    const auto [rows, inner, cols] = trace.coverage;
    if (n == 0 || rows == 0 || inner == 0 || cols == 0 || n % rows || n % inner || n % cols ||
        rows > std::numeric_limits<std::size_t>::max() / inner ||
        rows * inner > std::numeric_limits<std::size_t>::max() / cols)
        throw std::invalid_argument("coverage needs a valid uniform partition of the padded matrix");
    std::vector<bool> seen(rows * inner * cols);
    for (const auto& step : trace.supersteps) {
        if (step.flops.size() != trace.nodes)
            throw std::invalid_argument("coverage trace node count mismatch");
        std::vector<std::size_t> kernel_flops(trace.nodes);
        for (const auto& record : step.products) {
            const auto& p = record.product;
            if (record.node >= trace.nodes || p.i >= rows || p.k >= inner || p.j >= cols || p.m != n / rows ||
                p.inner != n / inner || p.n != n / cols)
                throw std::runtime_error("executed multiplication does not match its coverage block");
            const auto index = (p.i * inner + p.k) * cols + p.j;
            if (seen[index])
                throw std::runtime_error("duplicate executed (i,k,j) block");
            seen[index] = true;
            if (p.m > std::numeric_limits<std::size_t>::max() / p.n / p.inner / 2)
                throw std::invalid_argument("coverage FLOP count overflows");
            const auto work = 2 * p.m * p.n * p.inner;
            if (kernel_flops[record.node] > std::numeric_limits<std::size_t>::max() - work)
                throw std::invalid_argument("coverage node FLOP count overflows");
            kernel_flops[record.node] += work;
        }
        for (std::size_t id = 0; id < trace.nodes; ++id)
            if (kernel_flops[id] > step.flops[id])
                throw std::runtime_error("executed multiplication exceeds recorded node FLOPs");
    }
    if (std::find(seen.begin(), seen.end(), false) != seen.end())
        throw std::runtime_error("missing executed (i,k,j) block");
}

} // namespace dmm
