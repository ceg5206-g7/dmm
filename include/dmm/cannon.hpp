#pragma once

#include "dmm/algorithm.hpp"
#include "dmm/execution.hpp"

namespace dmm {

inline constexpr std::size_t GRID = 4;

struct Cannon {
    Matrix<float> matmul(const Matrix<float>& a, const Matrix<float>& b, Executor& executor) const;
};

static_assert(Algorithm<Cannon>);

struct Error {
    double max_absolute;
    double frobenius_relative;
};

Error validate(const Matrix<float>& result, const Matrix<double>& reference,
               double atol = 1e-4, double rtol = 1e-3);

} // namespace dmm
