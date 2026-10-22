#pragma once

#include "dmm/algorithm.hpp"
#include "dmm/execution.hpp"
#include "dmm/memory.hpp"
#include <concepts>
#include <cstddef>
#include <functional>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace dmm {

struct ElementRegion {
    std::size_t row, col, rows, cols;
};

// Payloads are counted in FP32 words; a multiply-add counts as two FLOPs.
struct MessageRecord {
    std::size_t source, destination, tag, words;
    std::optional<ElementRegion> region;
};

struct SuperstepRecord {
    std::string label;
    std::vector<std::size_t> flops; // Indexed by logical node.
    std::vector<MessageRecord> messages;
    std::vector<std::size_t> peak_words;
};

struct Trace {
    std::size_t nodes = 0;
    std::vector<SuperstepRecord> supersteps;
    std::vector<std::size_t> peak_words;
    std::size_t original_size = 0, padded_size = 0;
};

// h(r, s) is the larger of the words node r sends and receives in superstep s.
// Critical counts sum the per-superstep maxima over nodes, so they add across supersteps.
struct Counts {
    std::size_t supersteps = 0;
    std::size_t total_flops = 0, critical_flops = 0;
    std::size_t total_words = 0, critical_words = 0;
    std::size_t messages = 0;
    std::size_t peak_words = 0;

    Counts& operator+=(const Counts& other) noexcept;
    friend bool operator==(const Counts&, const Counts&) = default;
};

Counts count(const SuperstepRecord& superstep);
Counts count(const Trace& trace);
Counts count_native(const Trace& trace);

// Per-FLOP, per-word and per-superstep costs.
struct CostParameters {
    double w = 1, g = 0, l = 0;
};

// T = w * critical_flops + g * critical_words + l * supersteps.
double bsp_cost(const Counts& counts, const CostParameters& parameters);

namespace detail {
struct Message {
    std::size_t destination, tag;
    Buffer payload;
    std::optional<ElementRegion> region;
};

struct NodeState {
    std::vector<Message> outbox;
    std::map<std::pair<std::size_t, std::size_t>, Buffer> inbox; // (source, tag)
    std::size_t flops = 0;
    std::shared_ptr<MemoryLedger> memory = std::make_shared<MemoryLedger>();
};
} // namespace detail

// One logical node's handle for one superstep; only that node's task uses it.
class Node {
public:
    std::size_t id() const noexcept { return id_; }
    std::size_t nodes() const noexcept { return nodes_; }
    // Copies the payload; the destination receives it in the next superstep.
    void send(std::size_t destination, std::size_t tag, std::span<const float> payload,
              std::optional<ElementRegion> region = std::nullopt);
    // Moves out a message delivered at the end of the previous superstep.
    Buffer receive(std::size_t source, std::size_t tag, std::size_t words);
    Buffer buffer(std::size_t words = 0) const { return Buffer(state_.memory, std::vector<float>(words)); }
    void compute(std::size_t flops) noexcept { state_.flops += flops; }

private:
    friend class Bsp;
    Node(std::size_t id, std::size_t nodes, detail::NodeState& state) noexcept
        : id_(id), nodes_(nodes), state_(state) {}
    std::size_t id_, nodes_;
    detail::NodeState& state_;
};

// Logical nodes with private mailboxes. Communication is copied and counted,
// never delayed; costs come from the trace and CostParameters afterwards.
class Bsp {
public:
    explicit Bsp(std::size_t nodes);
    std::size_t nodes() const noexcept { return states_.size(); }
    Buffer buffer(std::size_t node, std::size_t words = 0) const;
    NodeBuffers buffers(std::size_t words = 0) const;
    std::size_t live_words(std::size_t node) const;
    bool owns(std::size_t node, const Buffer& buffer) const;
    // Runs body once per node, then delivers the sent messages. Every delivered
    // message must be received during the following superstep.
    void superstep(std::string label, Executor& executor, const std::function<void(Node&)>& body);
    // Checks that the last delivered messages were received and returns the trace.
    Trace finish();

private:
    std::vector<detail::NodeState> states_;
    Trace trace_;
};

struct Traced {
    Matrix<float> product;
    Trace trace;
};

template <typename T>
concept TracedAlgorithm = Algorithm<T> &&
    requires(const T& algorithm, const Matrix<float>& a, const Matrix<float>& b, Executor& executor) {
        { algorithm.run(a, b, executor) } -> std::same_as<Traced>;
    };

} // namespace dmm
