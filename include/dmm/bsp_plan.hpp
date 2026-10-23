#pragma once

#include "dmm/bsp.hpp"
#include "dmm/memory.hpp"
#include <array>
#include <utility>

namespace dmm {

inline constexpr std::array<const char*, 6> PHASE_NAMES{"input_conversion", "skew",      "replication",
                                                        "multiply",         "reduction", "output_conversion"};
struct Timing {
    double total = 0, compute = 0, copy = 0, synchronization = 0;
};
struct RunTimings {
    Timing end_to_end;
    std::array<Timing, 6> phases{};
};
struct BufferView {
    std::size_t buffer, offset, stride;
};

// Build once, seal, then execute without allocation. Views always name node-owned
// storage; a received copy is the only operation that reads another node's message.
// Host access/reset and execution must be serialized by the caller.
class BspPlan {
  public:
    explicit BspPlan(std::size_t nodes);
    std::size_t nodes() const noexcept { return memory_.size(); }
    bool owns(std::size_t node, std::size_t buffer) const { return owners_.at(buffer) == node; }
    std::size_t allocate(std::size_t node, std::size_t words);
    std::span<float> buffer(std::size_t id);
    std::size_t step(const std::string& label);
    void zero(std::size_t step, std::size_t node, std::size_t buffer);
    void copy(std::size_t step, std::size_t node, BufferView from, BufferView to, std::size_t rows,
              std::size_t cols);
    std::size_t send(std::size_t step, std::size_t source, std::size_t destination, std::size_t tag,
                     BufferView from, ElementRegion region);
    void receive(std::size_t step, std::size_t node, std::size_t message, BufferView to, bool add = false);
    void multiply_add(std::size_t step, std::size_t node, ProductRegion product, BufferView a, BufferView b,
                      BufferView c);
    void seal(std::size_t original_size, std::size_t padded_size, CoverageGrid coverage);
    RunTimings execute(Executor& executor);
    const Trace& trace() const;

  private:
    enum class Kind { zero, copy, receive, gemm, send };
    struct Operation {
        Kind kind;
        BufferView a{}, b{}, c{};
        std::size_t rows = 0, cols = 0, message = 0, record = 0;
        bool add = false;
        ProductRegion product{};
    };
    struct Step {
        std::size_t phase;
        std::vector<std::vector<Operation>> copies, kernels, sends;
    };
    struct Message {
        std::size_t step, source, destination, tag;
        ElementRegion region;
        std::size_t storage = 0;
        bool consumed = false, sent = false, received = false;
    };
    void building() const;
    void check_view(std::size_t node, BufferView view, std::size_t rows, std::size_t cols) const;
    std::span<float> view(BufferView value);
    std::vector<std::shared_ptr<detail::MemoryLedger>> memory_;
    std::vector<Buffer> buffers_;
    std::vector<std::size_t> owners_;
    std::vector<Step> steps_;
    std::vector<Message> messages_;
    std::vector<double> node_times_;
    Trace trace_;
    bool sealed_ = false, executed_ = false;
};

} // namespace dmm
