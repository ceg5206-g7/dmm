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
    peak_words = std::max(peak_words, other.peak_words);
    return *this;
}

Counts count(const SuperstepRecord& superstep) {
    const auto nodes = superstep.flops.size();
    std::vector<std::size_t> sent(nodes), received(nodes);
    Counts counts;
    for (auto words : superstep.peak_words) counts.peak_words = std::max(counts.peak_words, words);
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
    for (auto words : trace.peak_words) counts.peak_words = std::max(counts.peak_words, words);
    return counts;
}

Counts count_native(const Trace& trace) {
    Counts counts;
    for (const auto& step : trace.supersteps)
        if (step.label != "input_conversion" && step.label != "output_conversion") counts += count(step);
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

} // namespace dmm
