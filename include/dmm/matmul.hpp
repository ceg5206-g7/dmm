#pragma once
#include "dmm/matrix.hpp"
#include "dmm/plan.hpp"
#include <mutex>

namespace dmm {
struct MatmulConfig {
    std::size_t padded_size, P, c;
};
struct MatmulDefinition {
    std::vector<MatrixView> input_a_rows, input_b_rows, output_c_rows;
    CoverageGrid product_grid{};
};
using PlanBuilder = MatmulDefinition (*)(BspPlan&, const MatmulConfig&);
struct Traced {
    Matrix<float> product;
    Trace trace;
};

class PreparedMatmul {
  public:
    PreparedMatmul(Matrix<float> a, Matrix<float> b, std::size_t P, std::size_t c, PlanBuilder build);
    std::size_t size() const noexcept { return a_.size; }
    RunTimings run_into(std::span<float> output, Executor& executor);
    Trace trace() const;
    Traced run(Executor& executor);

  private:
    RunTimings execute_into(std::span<float> output, Executor& executor);
    Matrix<float> a_, b_;
    const MatmulConfig config_;
    BspPlan plan_;
    MatmulDefinition definition_;
    mutable std::mutex mutex_;
};
} // namespace dmm
