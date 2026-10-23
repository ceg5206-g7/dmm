#include "dmm/algorithms.hpp"
#include <string>

namespace dmm {
PlanBuilder select_builder(std::string_view layout) {
    if (layout == "cannon")
        return build_cannon;
    if (layout == "ring" || layout == "2.5d")
        throw std::invalid_argument("Layout " + std::string(layout) + " is not implemented");
    throw std::invalid_argument("unknown layout: " + std::string(layout));
}
Traced matmul_traced(const Matrix<float>& a, const Matrix<float>& b, Executor& executor,
                     std::string_view layout, std::size_t P, std::size_t c) {
    return PreparedMatmul(a, b, P, c, select_builder(layout)).run(executor);
}
Matrix<float> matmul(const Matrix<float>& a, const Matrix<float>& b, Executor& executor,
                     std::string_view layout, std::size_t P, std::size_t c) {
    return matmul_traced(a, b, executor, layout, P, c).product;
}
} // namespace dmm
