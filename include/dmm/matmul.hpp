#pragma once

#include "dmm/bsp_plan.hpp"
#include <mutex>

namespace dmm {

// Builder output: node-owned buffers start and finish as common row shards.
// Their capacity may be larger so an algorithm can reuse them as local tiles.
struct MatmulLayout {
    const std::size_t n, P, c;
    std::vector<std::size_t> a, b, result;
    CoverageGrid coverage{};
};
using PlanBuilder = void (*)(BspPlan&, MatmulLayout&);

// One lifecycle for every layout. A builder only allocates buffers and appends
// operations; host reset/gather, locking and timing boundaries live here.
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
    MatmulLayout layout_;
    BspPlan plan_;
    mutable std::mutex mutex_;
};

} // namespace dmm
