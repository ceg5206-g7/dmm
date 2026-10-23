#pragma once
#include "dmm/matrix.hpp"
#include "dmm/trace.hpp"
namespace dmm {
struct Error {
    double max_absolute, frobenius_relative;
};
Error validate(const Matrix<float>& result, const Matrix<double>& reference, double atol = 1e-4,
               double rtol = 1e-3);
void validate_coverage(const Trace& trace);
} // namespace dmm
