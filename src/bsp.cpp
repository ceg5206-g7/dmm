#include "dmm/bsp.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace dmm {

Counts& Counts::operator+=(const Counts& other) noexcept {
    supersteps += other.supersteps;
    total_flops += other.total_flops;
    critical_flops += other.critical_flops;
    total_words += other.total_words;
    critical_words += other.critical_words;
    messages += other.messages;
    return *this;
}

Counts count(const SuperstepRecord& superstep) {
    const auto nodes = superstep.flops.size();
    std::vector<std::size_t> sent(nodes), received(nodes);
    Counts counts;
    counts.supersteps = 1;
    counts.messages = superstep.messages.size();
    for (const auto& message : superstep.messages) {
        sent.at(message.source) += message.words;
        received.at(message.destination) += message.words;
        counts.total_words += message.words;
    }
    for (std::size_t node = 0; node < nodes; ++node) {
        counts.total_flops += superstep.flops[node];
        counts.critical_flops = std::max(counts.critical_flops, superstep.flops[node]);
        counts.critical_words = std::max({counts.critical_words, sent[node], received[node]});
    }
    return counts;
}

Counts count(const Trace& trace) {
    Counts counts;
    for (const auto& superstep : trace.supersteps) counts += count(superstep);
    return counts;
}

double bsp_cost(const Counts& counts, const CostParameters& parameters) {
    for (const auto value : {parameters.w, parameters.g, parameters.l})
        if (!std::isfinite(value) || value < 0)
            throw std::invalid_argument("BSP cost parameters must be finite and non-negative");
    return parameters.w * static_cast<double>(counts.critical_flops) +
           parameters.g * static_cast<double>(counts.critical_words) +
           parameters.l * static_cast<double>(counts.supersteps);
}

void Node::send(std::size_t destination, std::size_t tag, std::span<const float> payload) {
    if (destination >= nodes_) throw std::out_of_range("message destination is not a logical node");
    if (destination == id_) throw std::invalid_argument("a node keeps its own data instead of sending it");
    for (const auto& message : state_.outbox)
        if (message.destination == destination && message.tag == tag)
            throw std::logic_error("duplicate message destination and tag in one superstep");
    state_.outbox.push_back({destination, tag, {payload.begin(), payload.end()}});
}

std::vector<float> Node::receive(std::size_t source, std::size_t tag, std::size_t words) {
    if (source >= nodes_) throw std::out_of_range("message source is not a logical node");
    const auto message = state_.inbox.find({source, tag});
    if (message == state_.inbox.end())
        throw std::logic_error("no message was delivered from this source and tag");
    if (message->second.size() != words)
        throw std::logic_error("received payload length does not match the expected length");
    auto payload = std::move(message->second);
    state_.inbox.erase(message);
    return payload;
}

Bsp::Bsp(std::size_t nodes) : states_(nodes), trace_{nodes, {}} {
    if (nodes == 0) throw std::invalid_argument("BSP machine requires at least one logical node");
}

void Bsp::superstep(std::string label, Executor& executor, const std::function<void(Node&)>& body) {
    const auto nodes = states_.size();
    executor.for_each(nodes, [&](std::size_t id) {
        Node node(id, nodes, states_[id]);
        body(node);
    });
    for (const auto& state : states_)
        if (!state.inbox.empty())
            throw std::logic_error("superstep " + label + " left a delivered message unreceived");
    SuperstepRecord record{std::move(label), std::vector<std::size_t>(nodes), {}};
    for (std::size_t source = 0; source < nodes; ++source) {
        auto& state = states_[source];
        record.flops[source] = std::exchange(state.flops, 0);
        for (auto& message : state.outbox) {
            record.messages.push_back({source, message.destination, message.tag, message.payload.size()});
            states_[message.destination].inbox.emplace(std::pair{source, message.tag}, std::move(message.payload));
        }
        state.outbox.clear();
    }
    trace_.supersteps.push_back(std::move(record));
}

Trace Bsp::finish() {
    for (const auto& state : states_)
        if (!state.inbox.empty())
            throw std::logic_error("the final superstep left a delivered message unreceived");
    return std::exchange(trace_, Trace{states_.size(), {}});
}

} // namespace dmm
