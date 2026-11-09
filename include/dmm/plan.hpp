#pragma once

#include "dmm/detail/memory.hpp"
#include "dmm/execution.hpp"
#include "dmm/trace.hpp"
#include <initializer_list>
#include <limits>
#include <span>

namespace dmm {
class BspPlan;

// Storage handles are scoped to one nonmovable plan.
class BufferId {
  public:
    BufferId() = default;
    friend bool operator==(const BufferId&, const BufferId&) = default;

  private:
    friend class BspPlan;
    BufferId(const BspPlan* plan, std::size_t index) : plan_(plan), index_(index) {}
    const BspPlan* plan_ = nullptr;
    std::size_t index_ = 0;
};
struct MatrixView {
    BufferId buffer;
    std::size_t rows = 0, cols = 0, offset = 0, stride = 0;
    MatrixView submatrix(std::size_t row, std::size_t col, std::size_t height, std::size_t width) const;
};
struct GlobalPosition {
    std::size_t row = 0, col = 0;
};
struct PayloadPart {
    MatrixView view;
    GlobalPosition origin{}; // Trace coordinates; never used for matching.
};
struct BlockTriple {
    std::size_t i, k, j;
};

// A build-only handle for the latest superstep. Operations on each node must
// be declared in local/receive-copy, GEMM, then send order.
class Superstep {
  public:
    void add_zero(MatrixView target);
    void add_copy(MatrixView source, MatrixView destination);
    void add_gemm(MatrixView a, MatrixView b, MatrixView c, BlockTriple product);
    void add_send(std::size_t destination, MatrixView payload);
    void add_send(std::size_t destination, PayloadPart payload);
    void add_send(std::size_t destination, std::span<const PayloadPart> payload);
    void add_send(std::size_t destination, std::initializer_list<PayloadPart> payload);
    void add_receive(std::size_t source, MatrixView destination);
    void add_receive(std::size_t source, std::span<const MatrixView> destinations);
    void add_receive(std::size_t source, std::initializer_list<MatrixView> destinations);
    void add_receive_sum(std::size_t source, MatrixView destination);
    void add_receive_sum(std::size_t source, std::span<const MatrixView> destinations);
    void add_receive_sum(std::size_t source, std::initializer_list<MatrixView> destinations);

  private:
    friend class BspPlan;
    Superstep(BspPlan& plan, std::size_t index) : plan_(&plan), index_(index) {}
    BspPlan* plan_;
    std::size_t index_;
};

class BspPlan {
  public:
    explicit BspPlan(std::size_t nodes);
    BspPlan(const BspPlan&) = delete;
    BspPlan& operator=(const BspPlan&) = delete;
    std::size_t nodes() const noexcept { return memory_.size(); }
    BufferId allocate_buffer(std::size_t owner, std::size_t words);
    // Zero stride selects contiguous rows. Shape and offsets are checked here and at every operation.
    MatrixView matrix_view(BufferId buffer, std::size_t rows, std::size_t cols, std::size_t offset = 0,
                           std::size_t stride = 0) const;
    std::size_t owner(MatrixView view) const;
    std::span<float> buffer(BufferId id); // Host reset/gather; serialize with execute.
    // Host-written inputs stay live from the first superstep; host-read outputs
    // stay live through the last. Other payload is live only between its first
    // and last use (from step 0 when its first use reads it).
    void mark_host_input(BufferId id);
    void mark_host_output(BufferId id);
    Superstep add_superstep(Phase phase);
    // merge_delivery folds every superstep that neither sends nor computes into
    // its successor: no message leaves it, so its closing barrier is redundant.
    // Its receives and copies run first in the successor, under that label.
    void finalize(std::size_t original_size, std::size_t padded_size, CoverageGrid product_grid,
                  bool merge_delivery = false);
    RunTimings execute(Executor& executor);
    const Trace& trace() const;

  private:
    friend class Superstep;
    static constexpr auto none = std::numeric_limits<std::size_t>::max();
    enum class Kind { Zero, Copy, Receive, ReceiveSum, Gemm, Send };
    struct Operation {
        Kind kind;
        MatrixView a{}, b{}, c{};
        BlockTriple product{};
        std::size_t message = 0, record = 0;
    };
    struct Step {
        Phase phase;
        std::vector<std::vector<Operation>> copies, kernels, sends;
        std::vector<unsigned char> stages;
        std::vector<std::size_t> outgoing, incoming;
    };
    struct Message {
        std::size_t step, source, destination;
        std::vector<PayloadPart> parts;
        BufferId storage;
        std::size_t words = 0;
        bool sent = false, received = false;
    };
    struct Receive {
        std::size_t step, source, destination;
        std::vector<MatrixView> targets;
        std::size_t message = none;
    };
    void building() const;
    void current(std::size_t step) const;
    std::size_t check_buffer(BufferId buffer) const;
    std::size_t check_view(MatrixView view) const;
    void append(std::size_t step, std::size_t node, unsigned char stage, Operation operation);
    void add_receive(std::size_t step, std::size_t source, std::span<const MatrixView> targets, bool sum);
    bool overlaps(MatrixView a, MatrixView b) const;
    std::span<float> storage(MatrixView view);
    void merge_delivery_steps();
    void infer_lifetimes();
    std::vector<std::shared_ptr<detail::MemoryLedger>> memory_;
    std::shared_ptr<detail::MemoryLedger> idle_ = std::make_shared<detail::MemoryLedger>();
    std::vector<Buffer> buffers_;
    std::vector<std::size_t> owners_;
    // Per buffer: host flags (1 input, 2 output), message-slot flag and live range.
    std::vector<unsigned char> host_, slot_;
    std::vector<std::size_t> live_first_, live_last_;
    std::vector<std::vector<std::size_t>> live_starts_, live_ends_; // Buffer ids per step.
    std::vector<std::size_t> incoming_words_;
    std::vector<Step> steps_;
    std::vector<Message> messages_;
    std::vector<Receive> receives_;
    std::vector<double> node_times_;
    Trace trace_;
    bool finalized_ = false, executed_ = false;
};
} // namespace dmm
