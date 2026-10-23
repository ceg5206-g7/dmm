#pragma once
#include <cstddef>
namespace allocation_test {
// One process-wide counter also observes allocations on executor workers.
void start();
std::size_t stop();
class Watch {
  public:
    Watch() { start(); }
    ~Watch() { stop(); }
    std::size_t finish() { return stop(); }
};
} // namespace allocation_test
