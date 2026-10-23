#pragma once
#include "dmm/execution.hpp"
#include <string>

namespace dmm {
// Independent calibration; storage is allocated before internal timers.
double microbenchmark(const std::string& kind, std::size_t size, std::size_t iterations, std::size_t nodes,
                      Executor& executor);
} // namespace dmm
