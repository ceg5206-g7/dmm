#include "dmm/bsp_plan.hpp"
#include "dmm/gemm.hpp"
#include <chrono>
#include <limits>
#include <map>
#include <tuple>

namespace dmm {
namespace {
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }
void rectangle(std::span<const float> from, std::size_t lda, std::span<float> to, std::size_t ldb,
               std::size_t rows, std::size_t cols, bool add = false) {
    for (std::size_t row = 0; row < rows; ++row)
        if (add)
            for (std::size_t col = 0; col < cols; ++col) to[row * ldb + col] += from[row * lda + col];
        else std::copy_n(from.data() + row * lda, cols, to.data() + row * ldb);
}
}

BspPlan::BspPlan(std::size_t nodes) : node_times_(nodes), trace_{nodes, {}, {}} {
    if (!nodes) throw std::invalid_argument("prepared BSP needs positive nodes");
    for (std::size_t i = 0; i < nodes; ++i) memory_.push_back(std::make_shared<detail::MemoryLedger>());
}
void BspPlan::building() const {
    if (sealed_) throw std::logic_error("prepared plan is sealed");
}
std::size_t BspPlan::allocate(std::size_t node, std::size_t words) {
    building();
    auto ledger = memory_.at(node);
    buffers_.push_back(Buffer(std::move(ledger), std::vector<float>(words)));
    owners_.push_back(node);
    return buffers_.size() - 1;
}
std::span<float> BspPlan::buffer(std::size_t id) { return buffers_.at(id); }
std::span<float> BspPlan::view(BufferView v) { return buffer(v.buffer).subspan(v.offset); }
void BspPlan::check_view(std::size_t node, BufferView v, std::size_t rows, std::size_t cols) const {
    if (node >= memory_.size() || owners_.at(v.buffer) != node || !rows || !cols || v.offset > buffers_[v.buffer].size())
        throw std::invalid_argument("prepared operation must use its node's nonempty storage");
    detail::check_gemm_storage(rows, cols, v.stride, buffers_[v.buffer].size() - v.offset);
}
std::size_t BspPlan::step(const std::string& label) {
    building();
    const auto phase = std::find(PHASE_NAMES.begin(), PHASE_NAMES.end(), label);
    if (phase == PHASE_NAMES.end()) throw std::invalid_argument("unknown prepared phase");
    const auto nodes = memory_.size();
    steps_.push_back({static_cast<std::size_t>(phase - PHASE_NAMES.begin()),
                     std::vector<std::vector<Operation>>(nodes), std::vector<std::vector<Operation>>(nodes),
                     std::vector<std::vector<Operation>>(nodes)});
    trace_.supersteps.push_back({label, std::vector<std::size_t>(nodes), {}, std::vector<std::size_t>(nodes), {}});
    return steps_.size() - 1;
}
void BspPlan::zero(std::size_t s, std::size_t node, std::size_t id) {
    building();
    check_view(node, {id, 0, buffers_.at(id).size()}, 1, buffers_.at(id).size());
    steps_.at(s).copies.at(node).push_back({Kind::zero, {id, 0, 0}});
}
void BspPlan::copy(std::size_t s, std::size_t node, BufferView from, BufferView to,
                       std::size_t rows, std::size_t cols) {
    building(); check_view(node, from, rows, cols); check_view(node, to, rows, cols);
    if (from.buffer == to.buffer) throw std::invalid_argument("use a separate scratch buffer for in-place layout conversion");
    steps_.at(s).copies.at(node).push_back({Kind::copy, from, to, {}, rows, cols});
}
std::size_t BspPlan::send(std::size_t s, std::size_t src, std::size_t dst, std::size_t tag,
                             BufferView from, ElementRegion region) {
    building(); check_view(src, from, region.rows, region.cols);
    if (dst >= memory_.size() || dst == src || region.rows > std::numeric_limits<std::size_t>::max() / region.cols)
        throw std::invalid_argument("invalid prepared message destination/size");
    for (const auto& m : messages_)
        if (m.step == s && m.source == src && m.destination == dst && m.tag == tag)
            throw std::invalid_argument("duplicate prepared message");
    auto& step = steps_.at(s);
    const auto id = messages_.size(), record = trace_.supersteps[s].messages.size();
    step.sends.at(src).push_back({Kind::send, from, {}, {}, region.rows, region.cols, id, record});
    messages_.push_back({s, src, dst, tag, region});
    trace_.supersteps[s].messages.push_back({});
    return id;
}
void BspPlan::receive(std::size_t s, std::size_t node, std::size_t id, BufferView to, bool add) {
    building();
    auto& m = messages_.at(id);
    if (m.step + 1 != s || m.destination != node || m.consumed)
        throw std::invalid_argument("prepared receive must consume exactly one previous-step message");
    check_view(node, to, m.region.rows, m.region.cols);
    steps_.at(s).copies.at(node).push_back({Kind::receive, {}, to, {}, m.region.rows, m.region.cols, id, 0, add});
    m.consumed = true;
}
void BspPlan::multiply_add(std::size_t s, std::size_t node, ProductRegion p,
                               BufferView a, BufferView b, BufferView c) {
    building(); check_view(node, a, p.m, p.inner); check_view(node, b, p.inner, p.n); check_view(node, c, p.m, p.n);
    if (c.buffer == a.buffer || c.buffer == b.buffer || p.m > std::numeric_limits<std::size_t>::max() / p.n / p.inner / 2)
        throw std::invalid_argument("prepared GEMM has aliasing or overflowing work");
    const auto record = trace_.supersteps.at(s).products.size();
    steps_.at(s).kernels.at(node).push_back({Kind::gemm, a, b, c, 0, 0, 0, record, false, p});
    trace_.supersteps[s].products.push_back({});
}
void BspPlan::seal(std::size_t original, std::size_t padded, CoverageGrid grid) {
    building();
    if (!original || padded < original || steps_.empty()) throw std::invalid_argument("invalid prepared matrix dimensions/steps");
    if (!grid.row_parts || !grid.inner_parts || !grid.col_parts ||
        padded % grid.row_parts || padded % grid.inner_parts || padded % grid.col_parts)
        throw std::invalid_argument("coverage grid must partition the padded matrix");
    detail::elements<float>(padded);
    // Reuse each sender's same-size message slots after all previous receives finish.
    std::map<std::tuple<std::size_t, std::size_t, std::size_t>, std::size_t> cache;
    std::map<std::tuple<std::size_t, std::size_t, std::size_t>, std::size_t> ordinals;
    for (const auto& m : messages_) {
        const auto& r = m.region;
        if (!m.consumed || r.row > padded || r.rows > padded - r.row || r.col > padded || r.cols > padded - r.col)
            throw std::invalid_argument("undrained prepared message or region outside padded matrix");
    }
    for (auto& m : messages_) {
        const auto words = m.region.rows * m.region.cols;
        const auto ordinal = ordinals[{m.step, m.source, words}]++;
        const auto key = std::tuple{m.source, words, ordinal};
        if (!cache.contains(key)) cache[key] = allocate(m.source, words);
        m.storage = cache[key];
    }
    // Validate summed per-node work before entering an allocation-free execution.
    for (const auto& step : steps_)
        for (std::size_t node = 0; node < memory_.size(); ++node) {
            std::size_t work = 0;
            auto add = [&](std::size_t f) {
                if (f > std::numeric_limits<std::size_t>::max() - work) throw std::overflow_error("prepared FLOPs overflow");
                work += f;
            };
            for (const auto& op : step.kernels[node]) add(2 * op.product.m * op.product.n * op.product.inner);
            for (const auto& op : step.copies[node]) if (op.add) add(op.rows * op.cols);
        }
    trace_.original_size = original; trace_.padded_size = padded; trace_.coverage = grid;
    trace_.peak_words.resize(memory_.size());
    sealed_ = true;
}
RunTimings BspPlan::execute(Executor& executor) {
    if (!sealed_) throw std::logic_error("seal prepared plan before execution");
    executed_ = false;
    // A failed executor batch may leave delivered cache slots at their recipients.
    for (const auto& m : messages_)
        if (buffers_[m.storage].ledger_ != memory_[m.source]) buffers_[m.storage].rebind(memory_[m.source]);
    for (auto& ledger : memory_) ledger->peak = ledger->live;
    for (auto& m : messages_) { m.sent = false; m.received = false; }
    RunTimings result;
    const auto start = Clock::now();
    for (std::size_t s = 0; s < steps_.size(); ++s) {
        auto& step = steps_[s]; auto& record = trace_.supersteps[s];
        auto& timing = result.phases[step.phase];
        const auto step_start = Clock::now();
        std::fill(record.flops.begin(), record.flops.end(), 0);
        for (auto& ledger : memory_) ledger->step_peak = ledger->live;
        for (std::size_t batch = 0; batch < 3; ++batch) {
            auto task = [&](std::size_t node) {
                const auto local_start = Clock::now();
                const auto& ops = batch == 0 ? step.copies[node] : batch == 1 ? step.kernels[node] : step.sends[node];
                for (const auto& op : ops) {
                    if (op.kind == Kind::zero) {
                        auto data = buffer(op.a.buffer); std::fill(data.begin(), data.end(), 0.f);
                    } else if (op.kind == Kind::copy) rectangle(view(op.a), op.a.stride, view(op.b), op.b.stride, op.rows, op.cols);
                    else if (op.kind == Kind::receive) {
                        auto& m = messages_[op.message];
                        if (!m.sent) throw std::logic_error("prepared receive before send");
                        rectangle(buffer(m.storage), op.cols, view(op.b), op.b.stride, op.rows, op.cols, op.add);
                        if (op.add) record.flops[node] += op.rows * op.cols;
                        m.received = true;
                    } else if (op.kind == Kind::gemm) {
                        const auto& p = op.product;
                        gemm_add(p.m, p.n, p.inner, view(op.a), op.a.stride, view(op.b), op.b.stride, view(op.c), op.c.stride);
                        record.flops[node] += 2 * p.m * p.n * p.inner;
                        record.products[op.record] = {node, p};
                    } else {
                        auto& m = messages_[op.message];
                        rectangle(view(op.a), op.a.stride, buffer(m.storage), op.cols, op.rows, op.cols);
                        record.messages[op.record] = {node, m.destination, m.tag, op.rows * op.cols, m.region};
                        m.sent = true;
                    }
                }
                node_times_[node] = seconds(local_start);
            };
            // reference_wrapper keeps the std::function copied by Executor in its inline storage.
            executor.for_each(memory_.size(), std::ref(task));
            const auto critical = *std::max_element(node_times_.begin(), node_times_.end());
            (batch == 1 ? timing.compute : timing.copy) += critical;
            if (batch == 0 && s != 0)
                for (const auto& m : messages_) if (m.step + 1 == s) {
                    if (!m.received) throw std::logic_error("prepared message was not received");
                    buffers_[m.storage].rebind(memory_[m.source]);
                }
            if (batch == 2)
                for (const auto& m : messages_) if (m.step == s) buffers_[m.storage].rebind(memory_[m.destination]);
        }
        for (std::size_t node = 0; node < memory_.size(); ++node) record.peak_words[node] = memory_[node]->step_peak;
        timing.total += seconds(step_start);
    }
    result.end_to_end.total = seconds(start);
    for (auto& t : result.phases) {
        t.synchronization = std::max(0., t.total - t.compute - t.copy);
        result.end_to_end.compute += t.compute; result.end_to_end.copy += t.copy;
    }
    result.end_to_end.synchronization = std::max(0., result.end_to_end.total - result.end_to_end.compute - result.end_to_end.copy);
    for (std::size_t node = 0; node < memory_.size(); ++node) trace_.peak_words[node] = memory_[node]->peak;
    executed_ = true;
    return result;
}
const Trace& BspPlan::trace() const {
    if (!executed_) throw std::logic_error("prepared trace requires a successful execution");
    return trace_;
}

} // namespace dmm
