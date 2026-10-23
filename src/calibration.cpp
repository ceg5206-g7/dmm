#include "dmm/calibration.hpp"
#include "dmm/algorithm.hpp"
#include "dmm/gemm.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <limits>

namespace dmm {
namespace {
using Clock = std::chrono::steady_clock;
std::atomic<float> micro_sink{0};
double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
} // namespace
double microbenchmark(const std::string& kind, std::size_t size, std::size_t iterations, std::size_t nodes,
                      Executor& executor) {
    if ((kind != "compute" && kind != "copy" && kind != "synchronization") || !size || !iterations || !nodes)
        throw std::invalid_argument("invalid microbenchmark specification");
    const auto words = kind == "compute" ? detail::elements<float>(size) : size;
    std::vector<std::vector<float>> a(nodes, std::vector<float>(words, .01f)), b = a, c = a;
    std::vector<double> elapsed(nodes);
    auto task = [&](std::size_t node) {
        const auto start = Clock::now();
        for (std::size_t i = 0; i < iterations; ++i) {
            if (kind == "compute")
                gemm_add(size, size, size, a[node], size, b[node], size, c[node], size);
            else if (kind == "copy") {
                std::copy(a[node].begin(), a[node].end(), b[node].begin());
                std::atomic_signal_fence(std::memory_order_seq_cst);
                std::copy(b[node].begin(), b[node].end(), c[node].begin());
                std::atomic_signal_fence(std::memory_order_seq_cst);
            }
        }
        elapsed[node] = seconds(start);
        float checksum = 0;
        for (const auto value : c[node])
            checksum += value;
        micro_sink.store(checksum, std::memory_order_relaxed);
    };
    if (kind == "synchronization") {
        // The prepared protocol has three blocking executor batches per logical step.
        auto empty = [](std::size_t) {};
        const auto start = Clock::now();
        for (std::size_t i = 0; i < iterations; ++i)
            for (int batch = 0; batch < 3; ++batch)
                executor.for_each(nodes, std::ref(empty));
        return seconds(start);
    }
    executor.for_each(nodes, std::ref(task));
    return *std::max_element(elapsed.begin(), elapsed.end());
}
} // namespace dmm
