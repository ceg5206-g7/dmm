#include "dmm/plan.hpp"
#include "dmm/detail/gemm.hpp"
#include <chrono>

namespace dmm {
namespace {
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
void rectangle(std::span<const float> from, std::size_t lda, std::span<float> to, std::size_t ldb,
               std::size_t rows, std::size_t cols, bool add = false) {
    for (std::size_t row = 0; row < rows; ++row)
        if (add)
            for (std::size_t col = 0; col < cols; ++col)
                to[row * ldb + col] += from[row * lda + col];
        else
            std::copy_n(from.data() + row * lda, cols, to.data() + row * ldb);
}
} // namespace
RunTimings BspPlan::execute(Executor& executor) {
    if (!finalized_)
        throw std::logic_error("finalize plan before execution");
    executed_ = false;
    // Charge exactly the payload live before the first superstep: buffers whose
    // lifetime starts at step 0. Message slots and later-starting buffers are idle.
    // This also repairs the accounting after a failed batch.
    for (std::size_t id = 0; id < buffers_.size(); ++id) {
        const auto& ledger = slot_[id] || live_first_[id] ? idle_ : memory_[owners_[id]];
        if (buffers_[id].ledger_ != ledger)
            buffers_[id].rebind(ledger);
    }
    for (auto& ledger : memory_)
        ledger->peak = ledger->live;
    for (auto& message : messages_) {
        message.sent = false;
        message.received = false;
    }
    RunTimings result;
    const auto start = Clock::now();
    for (std::size_t s = 0; s < steps_.size(); ++s) {
        auto& step = steps_[s];
        auto& record = trace_.supersteps[s];
        auto& timing = result.phases[static_cast<std::size_t>(step.phase)];
        const auto step_start = Clock::now();
        std::fill(record.flops.begin(), record.flops.end(), 0);
        for (auto& ledger : memory_)
            ledger->step_peak = ledger->live;
        for (auto id : live_starts_[s])
            buffers_[id].rebind(memory_[owners_[id]]);
        for (unsigned char batch = 0; batch < 3; ++batch) {
            // A sender holds each outgoing payload from its send copy onwards.
            if (batch == 2)
                for (auto id : step.outgoing) {
                    const auto& message = messages_[id];
                    buffers_[message.storage.index_].rebind(memory_[message.source]);
                }
            auto task = [&](std::size_t node) {
                const auto local_start = Clock::now();
                const auto& ops = batch == 0   ? step.copies[node]
                                  : batch == 1 ? step.kernels[node]
                                               : step.sends[node];
                for (const auto& op : ops) {
                    if (op.kind == Kind::Zero) {
                        auto data = storage(op.a);
                        for (std::size_t row = 0; row < op.a.rows; ++row)
                            std::fill_n(data.data() + row * op.a.stride, op.a.cols, 0.f);
                    } else if (op.kind == Kind::Copy) {
                        rectangle(storage(op.a), op.a.stride, storage(op.b), op.b.stride, op.a.rows,
                                  op.a.cols);
                    } else if (op.kind == Kind::Gemm) {
                        gemm_add(op.a.rows, op.b.cols, op.a.cols, storage(op.a), op.a.stride, storage(op.b),
                                 op.b.stride, storage(op.c), op.c.stride);
                        record.flops[node] += 2 * op.a.rows * op.b.cols * op.a.cols;
                        record.products[op.record] = {
                            node,
                            {op.product.i, op.product.k, op.product.j, op.a.rows, op.b.cols, op.a.cols}};
                    } else if (op.kind == Kind::Send) {
                        auto& message = messages_[op.message];
                        auto data = buffer(message.storage);
                        auto& audit = record.messages[op.record];
                        std::size_t offset = 0;
                        for (std::size_t p = 0; p < message.parts.size(); ++p) {
                            const auto& part = message.parts[p];
                            rectangle(storage(part.view), part.view.stride, data.subspan(offset),
                                      part.view.cols, part.view.rows, part.view.cols);
                            audit.regions[p] = {part.origin.row, part.origin.col, part.view.rows,
                                                part.view.cols};
                            offset += part.view.rows * part.view.cols;
                        }
                        audit.source = node;
                        audit.destination = message.destination;
                        audit.words = offset;
                        message.sent = true;
                    } else {
                        const auto& receive = receives_[op.message];
                        auto& message = messages_[receive.message];
                        if (!message.sent)
                            throw std::logic_error("receive before actual send");
                        auto data = buffer(message.storage);
                        std::size_t offset = 0;
                        for (auto target : receive.targets) {
                            rectangle(data.subspan(offset), target.cols, storage(target), target.stride,
                                      target.rows, target.cols, op.kind == Kind::ReceiveSum);
                            const auto words = target.rows * target.cols;
                            if (op.kind == Kind::ReceiveSum)
                                record.flops[node] += words;
                            offset += words;
                        }
                        message.received = true;
                    }
                }
                node_times_[node] = seconds(local_start);
            };
            executor.for_each(nodes(), std::ref(task));
            const auto critical = *std::max_element(node_times_.begin(), node_times_.end());
            (batch == 1 ? timing.compute : timing.copy) += critical;
            // A receiver frees each delivered payload once it has been consumed.
            if (batch == 0)
                for (auto id : step.incoming) {
                    const auto& message = messages_[id];
                    if (!message.received)
                        throw std::logic_error("message was not received");
                    buffers_[message.storage.index_].rebind(idle_);
                }
        }
        // At the barrier both copies exist: senders still hold outgoing payloads
        // while receivers already hold incoming ones. Then ownership moves over.
        std::fill(incoming_words_.begin(), incoming_words_.end(), 0);
        for (auto id : step.outgoing)
            incoming_words_[messages_[id].destination] += buffers_[messages_[id].storage.index_].size();
        for (std::size_t node = 0; node < nodes(); ++node)
            memory_[node]->observe(incoming_words_[node]);
        for (auto id : step.outgoing) {
            const auto& message = messages_[id];
            buffers_[message.storage.index_].rebind(memory_[message.destination]);
        }
        for (auto id : live_ends_[s])
            buffers_[id].rebind(idle_);
        for (std::size_t node = 0; node < nodes(); ++node)
            record.peak_words[node] = memory_[node]->step_peak;
        timing.total += seconds(step_start);
    }
    result.end_to_end.total = seconds(start);
    for (auto& phase : result.phases) {
        phase.synchronization = std::max(0., phase.total - phase.compute - phase.copy);
        result.end_to_end.compute += phase.compute;
        result.end_to_end.copy += phase.copy;
    }
    result.end_to_end.synchronization =
        std::max(0., result.end_to_end.total - result.end_to_end.compute - result.end_to_end.copy);
    for (std::size_t node = 0; node < nodes(); ++node)
        trace_.peak_words[node] = memory_[node]->peak;
    executed_ = true;
    return result;
}
const Trace& BspPlan::trace() const {
    if (!executed_)
        throw std::logic_error("trace requires a successful execution");
    return trace_;
}
} // namespace dmm
