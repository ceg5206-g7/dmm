#pragma once
#include "dmm/matmul.hpp"
#include <string_view>

namespace dmm {
MatmulDefinition build_cannon(BspPlan& plan, const MatmulConfig& config);
MatmulDefinition build_ring(BspPlan& plan, const MatmulConfig& config);
MatmulDefinition build_two_point_five_d(BspPlan& plan, const MatmulConfig& config);
MatmulDefinition build_two_point_five_d_direct(BspPlan& plan, const MatmulConfig& config);
PlanBuilder select_builder(std::string_view layout);
Traced matmul_traced(const Matrix<float>& a, const Matrix<float>& b, Executor& executor,
                     std::string_view layout = "cannon", std::size_t P = 16, std::size_t c = 1);
Matrix<float> matmul(const Matrix<float>& a, const Matrix<float>& b, Executor& executor,
                     std::string_view layout = "cannon", std::size_t P = 16, std::size_t c = 1);
} // namespace dmm
