#include "dmm/prepared.hpp"
#include "dmm/gemm.hpp"
#include "dmm/layout.hpp"
#include "dmm/sharding.hpp"
#include <atomic>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <tuple>

namespace dmm {
namespace {
using Clock = std::chrono::steady_clock;
std::atomic<float> micro_sink{0};
double seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }
void rectangle(std::span<const float> from, std::size_t lda, std::span<float> to, std::size_t ldb,
               std::size_t rows, std::size_t cols, bool add = false) {
    for (std::size_t row = 0; row < rows; ++row)
        if (add)
            for (std::size_t col = 0; col < cols; ++col) to[row * ldb + col] += from[row * lda + col];
        else std::copy_n(from.data() + row * lda, cols, to.data() + row * ldb);
}
}

PreparedBsp::PreparedBsp(std::size_t nodes) : node_times_(nodes), trace_{nodes, {}, {}} {
    if (!nodes) throw std::invalid_argument("prepared BSP needs positive nodes");
    for (std::size_t i = 0; i < nodes; ++i) memory_.push_back(std::make_shared<detail::MemoryLedger>());
}
void PreparedBsp::building() const {
    if (sealed_) throw std::logic_error("prepared plan is sealed");
}
std::size_t PreparedBsp::allocate(std::size_t node, std::size_t words) {
    building();
    auto ledger = memory_.at(node);
    buffers_.push_back(Buffer(std::move(ledger), std::vector<float>(words)));
    owners_.push_back(node);
    return buffers_.size() - 1;
}
std::span<float> PreparedBsp::buffer(std::size_t id) { return buffers_.at(id); }
std::span<float> PreparedBsp::view(BufferView v) { return buffer(v.buffer).subspan(v.offset); }
void PreparedBsp::check_view(std::size_t node, BufferView v, std::size_t rows, std::size_t cols) const {
    if (node >= memory_.size() || owners_.at(v.buffer) != node || !rows || !cols || v.offset > buffers_[v.buffer].size())
        throw std::invalid_argument("prepared operation must use its node's nonempty storage");
    detail::check_gemm_storage(rows, cols, v.stride, buffers_[v.buffer].size() - v.offset);
}
std::size_t PreparedBsp::step(const std::string& label) {
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
void PreparedBsp::zero(std::size_t s, std::size_t node, std::size_t id) {
    building();
    check_view(node, {id, 0, buffers_.at(id).size()}, 1, buffers_.at(id).size());
    steps_.at(s).copies.at(node).push_back({Kind::zero, {id, 0, 0}});
}
void PreparedBsp::copy(std::size_t s, std::size_t node, BufferView from, BufferView to,
                       std::size_t rows, std::size_t cols) {
    building(); check_view(node, from, rows, cols); check_view(node, to, rows, cols);
    if (from.buffer == to.buffer) throw std::invalid_argument("use a separate scratch buffer for in-place layout conversion");
    steps_.at(s).copies.at(node).push_back({Kind::copy, from, to, {}, rows, cols});
}
std::size_t PreparedBsp::send(std::size_t s, std::size_t src, std::size_t dst, std::size_t tag,
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
    messages_.push_back({s, src, dst, tag, from, region});
    trace_.supersteps[s].messages.push_back({});
    return id;
}
void PreparedBsp::receive(std::size_t s, std::size_t node, std::size_t id, BufferView to, bool add) {
    building();
    auto& m = messages_.at(id);
    if (m.step + 1 != s || m.destination != node || m.consumed)
        throw std::invalid_argument("prepared receive must consume exactly one previous-step message");
    check_view(node, to, m.region.rows, m.region.cols);
    steps_.at(s).copies.at(node).push_back({Kind::receive, {}, to, {}, m.region.rows, m.region.cols, id, 0, add});
    m.consumed = true;
}
void PreparedBsp::multiply_add(std::size_t s, std::size_t node, ProductRegion p,
                               BufferView a, BufferView b, BufferView c) {
    building(); check_view(node, a, p.m, p.inner); check_view(node, b, p.inner, p.n); check_view(node, c, p.m, p.n);
    if (c.buffer == a.buffer || c.buffer == b.buffer || p.m > std::numeric_limits<std::size_t>::max() / p.n / p.inner / 2)
        throw std::invalid_argument("prepared GEMM has aliasing or overflowing work");
    const auto record = trace_.supersteps.at(s).products.size();
    steps_.at(s).kernels.at(node).push_back({Kind::gemm, a, b, c, 0, 0, 0, record, false, p});
    trace_.supersteps[s].products.push_back({});
}
void PreparedBsp::seal(std::size_t original, std::size_t padded, CoverageGrid grid) {
    building();
    if (!original || padded < original || steps_.empty()) throw std::invalid_argument("invalid prepared matrix dimensions/steps");
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
RunTimings PreparedBsp::execute(Executor& executor) {
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
const Trace& PreparedBsp::trace() const {
    if (!executed_) throw std::logic_error("prepared trace requires a successful execution");
    return trace_;
}

PreparedCannon::PreparedCannon(Matrix<float> a, Matrix<float> b, std::size_t P, std::size_t c)
    : a_(std::move(a)), b_(std::move(b)), P_(P), n_(padded_size(a_.size, P)), q_(grid_size(P, c)), plan_(P) {
    if (c != 1 || a_.size != b_.size || a_.values.size() != detail::elements<float>(a_.size) ||
        b_.values.size() != detail::elements<float>(b_.size)) throw std::invalid_argument("invalid prepared Cannon inputs/grid");
    for (const auto* input : {&a_, &b_})
        for (auto value : input->values) if (!std::isfinite(value)) throw std::invalid_argument("inputs must be finite");
    const auto block = n_ / q_, height = n_ / P_, words = block * block;
    for (std::size_t node = 0; node < P_; ++node) {
        left_.push_back(plan_.allocate(node, words)); right_.push_back(plan_.allocate(node, words));
        result_.push_back(plan_.allocate(node, words));
        if (P_ > 1) scratch_.push_back(plan_.allocate(node, height * block));
    }
    auto v = [&](std::size_t id) { return BufferView{id, 0, block}; };
    auto convert = [&](const std::vector<std::size_t>& ids, std::size_t tag, bool input) {
        const auto send_step = plan_.step(input ? "input_conversion" : "output_conversion");
        const auto recv_step = plan_.step(input ? "input_conversion" : "output_conversion");
        for (std::size_t source = 0; source < P_; ++source) {
            const auto i = source / q_, j = source % q_;
            // Save the retained rectangle before overwriting the same row/tile buffer.
            const BufferView row{ids[source], j * block, n_};
            const BufferView tile{ids[source], (source * height - i * block) * block, block};
            const BufferView scratch{scratch_[source], 0, block};
            plan_.copy(recv_step, source, input ? row : tile, scratch, height, block);
            plan_.copy(recv_step, source, scratch, input ? tile : row, height, block);
        }
        for (std::size_t row_node = 0; row_node < P_; ++row_node) {
            const auto i = row_node / q_;
            for (std::size_t j = 0; j < q_; ++j) {
                const auto tile_node = i * q_ + j;
                if (row_node == tile_node) continue;
                const BufferView row{ids[row_node], j * block, n_};
                const BufferView tile{ids[tile_node], (row_node * height - i * block) * block, block};
                const auto source = input ? row_node : tile_node, dest = input ? tile_node : row_node;
                const auto msg = plan_.send(send_step, source, dest, tag, input ? row : tile,
                                            {row_node * height, j * block, height, block});
                plan_.receive(recv_step, dest, msg, input ? tile : row);
            }
        }
    };
    if (P_ > 1) { convert(left_, 0, true); convert(right_, 1, true); }
    const auto skew = plan_.step("skew");
    for (std::size_t node = 0; node < P_; ++node) plan_.zero(P_ > 1 ? 0 : skew, node, result_[node]);
    const auto none = std::numeric_limits<std::size_t>::max();
    std::vector<std::size_t> pending_a(P_, none), pending_b(P_, none);
    auto at = [&](std::size_t i, std::size_t j) { return (i % q_) * q_ + j % q_; };
    for (std::size_t node = 0; node < P_; ++node) {
        const auto i = node / q_, j = node % q_;
        if (i) { const auto dest = at(i, j + q_ - i);
            pending_a[dest] = plan_.send(skew, node, dest, 0, v(left_[node]), {i * block, j * block, block, block}); }
        if (j) { const auto dest = at(i + q_ - j, j);
            pending_b[dest] = plan_.send(skew, node, dest, 1, v(right_[node]), {i * block, j * block, block, block}); }
    }
    for (std::size_t round = 0; round < q_; ++round) {
        const auto step = plan_.step("multiply");
        for (std::size_t node = 0; node < P_; ++node) {
            if (pending_a[node] != none) plan_.receive(step, node, pending_a[node], v(left_[node]));
            if (pending_b[node] != none) plan_.receive(step, node, pending_b[node], v(right_[node]));
            const auto i = node / q_, j = node % q_, k = (i + j + round) % q_;
            plan_.multiply_add(step, node, {i, k, j, block, block, block}, v(left_[node]), v(right_[node]), v(result_[node]));
        }
        if (round + 1 < q_) for (std::size_t node = 0; node < P_; ++node) {
            const auto i = node / q_, j = node % q_, k = (i + j + round) % q_;
            const auto dest_a = at(i, j + q_ - 1), dest_b = at(i + q_ - 1, j);
            pending_a[dest_a] = plan_.send(step, node, dest_a, 0, v(left_[node]), {i * block, k * block, block, block});
            pending_b[dest_b] = plan_.send(step, node, dest_b, 1, v(right_[node]), {k * block, j * block, block, block});
        }
    }
    if (P_ > 1) convert(result_, 2, false);
    plan_.seal(a_.size, n_, {q_, q_, q_});
}
RunTimings PreparedCannon::run_into(std::span<float> output, Executor& executor) {
    std::lock_guard lock(mutex_);
    if (output.size() != a_.values.size()) throw std::invalid_argument("prepared output size mismatch");
    const auto height = n_ / P_;
    for (std::size_t node = 0; node < P_; ++node)
        for (auto pair : {std::pair{&a_, left_[node]}, std::pair{&b_, right_[node]}}) {
            auto rows = plan_.buffer(pair.second);
            std::fill(rows.begin(), rows.end(), 0.f);
            for (std::size_t row = 0; row < height && node * height + row < a_.size; ++row)
                std::copy_n(pair.first->values.data() + (node * height + row) * a_.size, a_.size,
                            rows.data() + row * n_);
        }
    auto timing = plan_.execute(executor);
    for (std::size_t node = 0; node < P_; ++node) {
        auto rows = plan_.buffer(result_[node]);
        for (std::size_t row = 0; row < height && node * height + row < a_.size; ++row)
            std::copy_n(rows.data() + row * n_, a_.size, output.data() + (node * height + row) * a_.size);
    }
    return timing;
}
Trace PreparedCannon::trace() const { std::lock_guard lock(mutex_); return plan_.trace(); }

double microbenchmark(const std::string& kind, std::size_t size, std::size_t iterations,
                      std::size_t nodes, Executor& executor) {
    if ((kind != "compute" && kind != "copy" && kind != "synchronization") || !size || !iterations || !nodes)
        throw std::invalid_argument("invalid microbenchmark specification");
    const auto words = kind == "compute" ? detail::elements<float>(size) : size;
    std::vector<std::vector<float>> a(nodes, std::vector<float>(words, .01f)), b = a, c = a;
    std::vector<double> elapsed(nodes);
    auto task = [&](std::size_t node) {
        const auto start = Clock::now();
        for (std::size_t i = 0; i < iterations; ++i) {
            if (kind == "compute") gemm_add(size, size, size, a[node], size, b[node], size, c[node], size);
            else if (kind == "copy") {
                std::copy(a[node].begin(), a[node].end(), b[node].begin());
                std::atomic_signal_fence(std::memory_order_seq_cst);
                std::copy(b[node].begin(), b[node].end(), c[node].begin());
                std::atomic_signal_fence(std::memory_order_seq_cst);
            }
        }
        elapsed[node] = seconds(start);
        float checksum = 0;
        for (const auto value : c[node]) checksum += value;
        micro_sink.store(checksum, std::memory_order_relaxed);
    };
    if (kind == "synchronization") {
        // The prepared protocol has three blocking executor batches per logical step.
        auto empty = [](std::size_t) {};
        const auto start = Clock::now();
        for (std::size_t i = 0; i < iterations; ++i)
            for (int batch = 0; batch < 3; ++batch) executor.for_each(nodes, std::ref(empty));
        return seconds(start);
    }
    executor.for_each(nodes, std::ref(task));
    return *std::max_element(elapsed.begin(), elapsed.end());
}
} // namespace dmm
