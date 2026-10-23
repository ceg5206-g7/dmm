#pragma once

#include "dmm/algorithm.hpp"
#include "dmm/matmul.hpp"
#include "dmm/execution.hpp"
#include "dmm/layout.hpp"

namespace dmm {

// Common row-shard route with padding and conversions; the native body has
// one skew superstep followed by q multiply supersteps on P=q*q logical nodes.
void build_cannon(BspPlan& plan, MatmulLayout& layout);

// Compatibility facade; new layouts need only a build function.
struct Cannon {
    std::size_t P = 16;
    std::size_t c = 1;
    Matrix<float> matmul(const Matrix<float>& a, const Matrix<float>& b, Executor& executor) const;
    Traced run(const Matrix<float>& a, const Matrix<float>& b, Executor& executor) const;
};

static_assert(TracedAlgorithm<Cannon>);

struct Error {
    double max_absolute;
    double frobenius_relative;
};

Error validate(const Matrix<float>& result, const Matrix<double>& reference,
               double atol = 1e-4, double rtol = 1e-3);

} // namespace dmm
