#pragma once

#include "dmm/algorithm.hpp"
#include "dmm/bsp.hpp"
#include "dmm/execution.hpp"
#include "dmm/layout.hpp"

namespace dmm {

inline constexpr std::size_t GRID = 4;

// Common row-shard route with padding and conversions; the native body has
// one skew superstep followed by q multiply supersteps on P=q*q logical nodes.
// GRID is retained only as the default-grid compatibility constant.
struct Cannon {
    std::size_t P = 16;
    std::size_t c = 1;
    Matrix<float> matmul(const Matrix<float>& a, const Matrix<float>& b, Executor& executor) const;
    Traced run_native(const Matrix<float>& a, const Matrix<float>& b, Executor& executor) const;
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
