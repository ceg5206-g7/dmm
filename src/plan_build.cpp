#include "dmm/plan.hpp"
#include "dmm/detail/gemm.hpp"
#include "dmm/matrix.hpp"
#include <algorithm>
#include <map>
#include <tuple>

namespace dmm {
namespace {
std::size_t checked_add(std::size_t a, std::size_t b) {
    if (b > std::numeric_limits<std::size_t>::max() - a)
        throw std::overflow_error("plan size/work overflow");
    return a + b;
}
} // namespace
MatrixView MatrixView::submatrix(std::size_t row, std::size_t col, std::size_t height,
                                 std::size_t width) const {
    if (!height || !width || row > rows || height > rows - row || col > cols || width > cols - col ||
        (stride && row > std::numeric_limits<std::size_t>::max() / stride))
        throw std::invalid_argument("submatrix must fit inside its parent view");
    return {buffer, height, width, checked_add(offset, checked_add(row * stride, col)), stride};
}
BspPlan::BspPlan(std::size_t nodes) : node_times_(nodes), trace_{nodes, {}, {}} {
    if (!nodes)
        throw std::invalid_argument("plan needs positive logical nodes");
    for (std::size_t i = 0; i < nodes; ++i)
        memory_.push_back(std::make_shared<detail::MemoryLedger>());
}
void BspPlan::building() const {
    if (finalized_)
        throw std::logic_error("plan is finalized");
}
void BspPlan::current(std::size_t step) const {
    building();
    if (steps_.empty() || step != steps_.size() - 1)
        throw std::logic_error("only the latest superstep accepts operations");
}
std::size_t BspPlan::check_buffer(BufferId id) const {
    if (id.plan_ != this || id.index_ >= buffers_.size())
        throw std::invalid_argument("buffer belongs to another plan or is invalid");
    return id.index_;
}
std::size_t BspPlan::check_view(MatrixView v) const {
    const auto id = check_buffer(v.buffer);
    if (!v.rows || !v.cols || v.offset > buffers_[id].size())
        throw std::invalid_argument("matrix view must be nonempty and fit its storage");
    detail::check_gemm_storage(v.rows, v.cols, v.stride, buffers_[id].size() - v.offset);
    return owners_[id];
}
std::size_t BspPlan::owner(MatrixView v) const { return check_view(v); }
BufferId BspPlan::allocate_buffer(std::size_t node, std::size_t words) {
    building();
    auto ledger = memory_.at(node);
    buffers_.push_back(Buffer(std::move(ledger), std::vector<float>(words)));
    owners_.push_back(node);
    return {this, buffers_.size() - 1};
}
MatrixView BspPlan::matrix_view(BufferId buffer, std::size_t rows, std::size_t cols, std::size_t offset,
                                std::size_t stride) const {
    MatrixView view{buffer, rows, cols, offset, stride ? stride : cols};
    check_view(view);
    return view;
}
std::span<float> BspPlan::buffer(BufferId id) { return buffers_[check_buffer(id)]; }
std::span<float> BspPlan::storage(MatrixView v) { return buffer(v.buffer).subspan(v.offset); }
bool BspPlan::overlaps(MatrixView a, MatrixView b) const {
    if (a.buffer != b.buffer)
        return false;
    std::size_t i = 0, j = 0;
    while (i < a.rows && j < b.rows) {
        const auto x = a.offset + i * a.stride, y = b.offset + j * b.stride;
        if (x < y + b.cols && y < x + a.cols)
            return true;
        if (x + a.cols <= y)
            ++i;
        else
            ++j;
    }
    return false;
}
Superstep BspPlan::add_superstep(Phase phase) {
    building();
    const auto p = static_cast<std::size_t>(phase), n = nodes();
    if (p >= PHASE_NAMES.size())
        throw std::invalid_argument("unknown phase");
    steps_.push_back({phase,
                      std::vector<std::vector<Operation>>(n),
                      std::vector<std::vector<Operation>>(n),
                      std::vector<std::vector<Operation>>(n),
                      std::vector<unsigned char>(n),
                      {},
                      {}});
    trace_.supersteps.push_back(
        {PHASE_NAMES[p], std::vector<std::size_t>(n), {}, std::vector<std::size_t>(n), {}});
    return {*this, steps_.size() - 1};
}
void BspPlan::append(std::size_t s, std::size_t node, unsigned char stage, Operation op) {
    current(s);
    auto& step = steps_[s];
    if (stage < step.stages.at(node))
        throw std::logic_error("node operations must be copies, GEMMs, then sends");
    auto& ops = stage == 0 ? step.copies[node] : stage == 1 ? step.kernels[node] : step.sends[node];
    ops.push_back(op);
    step.stages[node] = stage;
}
void Superstep::add_zero(MatrixView target) {
    const auto node = plan_->check_view(target);
    plan_->append(index_, node, 0, {BspPlan::Kind::Zero, target});
}
void Superstep::add_copy(MatrixView source, MatrixView destination) {
    const auto node = plan_->check_view(source);
    if (plan_->check_view(destination) != node || source.rows != destination.rows ||
        source.cols != destination.cols || plan_->overlaps(source, destination))
        throw std::invalid_argument("copy needs equal nonoverlapping views on one node");
    plan_->append(index_, node, 0, {BspPlan::Kind::Copy, source, destination});
}
void Superstep::add_gemm(MatrixView a, MatrixView b, MatrixView c, BlockTriple product) {
    const auto node = plan_->check_view(a);
    if (plan_->check_view(b) != node || plan_->check_view(c) != node || a.cols != b.rows ||
        a.rows != c.rows || b.cols != c.cols || plan_->overlaps(a, c) || plan_->overlaps(b, c) ||
        a.rows > std::numeric_limits<std::size_t>::max() / b.cols / a.cols / 2)
        throw std::invalid_argument("GEMM needs compatible nonaliasing local views and finite work counts");
    const auto record = plan_->trace_.supersteps.at(index_).products.size();
    plan_->append(index_, node, 1, {BspPlan::Kind::Gemm, a, b, c, product, 0, record});
    plan_->trace_.supersteps[index_].products.push_back({});
}
void Superstep::add_send(std::size_t destination, MatrixView payload) {
    add_send(destination, PayloadPart{payload, {}});
}
void Superstep::add_send(std::size_t destination, PayloadPart payload) {
    add_send(destination, std::span(&payload, 1));
}
void Superstep::add_send(std::size_t destination, std::initializer_list<PayloadPart> parts) {
    add_send(destination, std::span(parts.begin(), parts.size()));
}
void Superstep::add_send(std::size_t destination, std::span<const PayloadPart> parts) {
    plan_->current(index_);
    if (parts.empty())
        throw std::invalid_argument("message needs at least one payload component");
    const auto source = plan_->check_view(parts.front().view);
    if (destination >= plan_->nodes() || destination == source)
        throw std::invalid_argument("invalid message destination");
    std::size_t words = 0;
    for (const auto& part : parts) {
        if (plan_->check_view(part.view) != source)
            throw std::invalid_argument("all payload components need the same owner");
        words = checked_add(words, part.view.rows * part.view.cols);
    }
    for (auto id : plan_->steps_[index_].outgoing)
        if (plan_->messages_[id].source == source && plan_->messages_[id].destination == destination)
            throw std::invalid_argument(
                "one logical message per node pair and superstep; bundle its components");
    const auto id = plan_->messages_.size(), record = plan_->trace_.supersteps[index_].messages.size();
    plan_->append(index_, source, 2, {BspPlan::Kind::Send, {}, {}, {}, {}, id, record});
    plan_->messages_.push_back({index_, source, destination, {parts.begin(), parts.end()}, {}, words});
    plan_->steps_[index_].outgoing.push_back(id);
    plan_->trace_.supersteps[index_].messages.push_back({0, 0, 0, std::vector<ElementRegion>(parts.size())});
}
void Superstep::add_receive(std::size_t source, MatrixView target) {
    add_receive(source, std::span(&target, 1));
}
void Superstep::add_receive(std::size_t source, std::initializer_list<MatrixView> targets) {
    add_receive(source, std::span(targets.begin(), targets.size()));
}
void Superstep::add_receive(std::size_t source, std::span<const MatrixView> targets) {
    plan_->add_receive(index_, source, targets, false);
}
void Superstep::add_receive_sum(std::size_t source, MatrixView target) {
    add_receive_sum(source, std::span(&target, 1));
}
void Superstep::add_receive_sum(std::size_t source, std::initializer_list<MatrixView> targets) {
    add_receive_sum(source, std::span(targets.begin(), targets.size()));
}
void Superstep::add_receive_sum(std::size_t source, std::span<const MatrixView> targets) {
    plan_->add_receive(index_, source, targets, true);
}
void BspPlan::add_receive(std::size_t s, std::size_t source, std::span<const MatrixView> targets, bool sum) {
    current(s);
    if (targets.empty())
        throw std::invalid_argument("receive needs at least one component");
    const auto destination = check_view(targets.front());
    if (!s || source >= nodes() || source == destination)
        throw std::invalid_argument("receive needs a different source in the previous superstep");
    for (std::size_t i = 0; i < targets.size(); ++i) {
        if (check_view(targets[i]) != destination)
            throw std::invalid_argument("receive components need the same owner");
        for (std::size_t j = 0; j < i; ++j)
            if (overlaps(targets[i], targets[j]))
                throw std::invalid_argument("receive components must not overlap");
    }
    for (const auto& r : receives_)
        if (r.step == s && r.source == source && r.destination == destination)
            throw std::invalid_argument("message can only be received once");
    const auto id = receives_.size();
    append(s, destination, 0, {sum ? Kind::ReceiveSum : Kind::Receive, {}, {}, {}, {}, id});
    receives_.push_back({s, source, destination, {targets.begin(), targets.end()}});
}
void BspPlan::merge_delivery_steps() {
    // Walk backwards so chains of delivery-only steps collapse into one successor.
    for (auto s = steps_.size(); s-- > 1;) {
        auto& step = steps_[s - 1];
        auto& next = steps_[s];
        bool idle = step.outgoing.empty();
        for (std::size_t node = 0; idle && node < nodes(); ++node) {
            idle = step.kernels[node].empty() && step.sends[node].empty();
            for (const auto& op : step.copies[node])
                idle = idle && op.kind != Kind::ReceiveSum;
            // The successor cannot receive: nothing was sent to it. Keep a
            // malformed schedule unmerged so finalize still rejects it.
            for (const auto& op : next.copies[node])
                idle = idle && op.kind != Kind::Receive && op.kind != Kind::ReceiveSum;
        }
        if (!idle)
            continue;
        for (std::size_t node = 0; node < nodes(); ++node) {
            next.copies[node].insert(next.copies[node].begin(), step.copies[node].begin(),
                                     step.copies[node].end());
            next.stages[node] = std::max(next.stages[node], step.stages[node]);
        }
        steps_.erase(steps_.begin() + static_cast<std::ptrdiff_t>(s - 1));
        trace_.supersteps.erase(trace_.supersteps.begin() + static_cast<std::ptrdiff_t>(s - 1));
        for (auto& m : messages_)
            if (m.step >= s)
                --m.step;
        for (auto& r : receives_)
            if (r.step >= s)
                --r.step;
    }
}
void BspPlan::finalize(std::size_t original, std::size_t padded, CoverageGrid grid, bool merge_delivery) {
    building();
    if (merge_delivery)
        merge_delivery_steps();
    if (!original || padded < original || steps_.empty() || !grid.row_parts || !grid.inner_parts ||
        !grid.col_parts || padded % grid.row_parts || padded % grid.inner_parts || padded % grid.col_parts)
        throw std::invalid_argument("invalid matrix dimensions, phases or coverage partition");
    detail::elements<float>(padded);
    std::map<std::tuple<std::size_t, std::size_t, std::size_t>, std::size_t> edges;
    for (std::size_t id = 0; id < messages_.size(); ++id) {
        const auto& m = messages_[id];
        edges.emplace(std::tuple{m.step, m.source, m.destination}, id);
        for (const auto& part : m.parts)
            if (part.origin.row > padded || part.view.rows > padded - part.origin.row ||
                part.origin.col > padded || part.view.cols > padded - part.origin.col)
                throw std::invalid_argument("payload region outside padded matrix");
    }
    std::vector<std::size_t> resolved(receives_.size()), consumed(messages_.size());
    for (std::size_t i = 0; i < receives_.size(); ++i) {
        const auto& r = receives_[i];
        const auto it = edges.find({r.step - 1, r.source, r.destination});
        if (it == edges.end())
            throw std::invalid_argument("receive has no matching previous-step send");
        const auto id = it->second;
        const auto& m = messages_[id];
        if (m.parts.size() != r.targets.size() || consumed[id]++)
            throw std::invalid_argument("message component count or receive multiplicity mismatch");
        for (std::size_t p = 0; p < m.parts.size(); ++p)
            if (m.parts[p].view.rows != r.targets[p].rows || m.parts[p].view.cols != r.targets[p].cols)
                throw std::invalid_argument("send/receive component shapes must match in order");
        resolved[i] = id;
    }
    for (auto n : consumed)
        if (n != 1)
            throw std::invalid_argument("every message needs one next-step receive");
    for (const auto& step : steps_)
        for (std::size_t node = 0; node < nodes(); ++node) {
            std::size_t work = 0;
            for (const auto& op : step.kernels[node])
                work = checked_add(work, 2 * op.a.rows * op.b.cols * op.a.cols);
            for (const auto& op : step.copies[node])
                if (op.kind == Kind::ReceiveSum)
                    for (auto target : receives_[op.message].targets)
                        work = checked_add(work, target.rows * target.cols);
        }
    std::map<std::tuple<std::size_t, std::size_t, std::size_t>, BufferId> cache;
    std::map<std::tuple<std::size_t, std::size_t, std::size_t>, std::size_t> ordinals;
    for (auto& m : messages_) {
        const auto ordinal = ordinals[{m.step, m.source, m.words}]++;
        const auto key = std::tuple{m.source, m.words, ordinal};
        if (!cache.contains(key))
            cache.emplace(key, allocate_buffer(m.source, m.words));
        m.storage = cache.at(key);
    }
    for (std::size_t i = 0; i < receives_.size(); ++i) {
        receives_[i].message = resolved[i];
        steps_[receives_[i].step].incoming.push_back(resolved[i]);
    }
    trace_.original_size = original;
    trace_.padded_size = padded;
    trace_.coverage = grid;
    trace_.peak_words.resize(nodes());
    finalized_ = true;
}
} // namespace dmm
