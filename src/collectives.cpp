#include "dmm/collectives.hpp"
#include <algorithm>
#include <bit>
#include <stdexcept>

namespace dmm {
namespace {
struct TreeNode {
    bool member = false;
    std::size_t depth = 0, parent = 0, words = 0;
    std::vector<std::size_t> children;
};

std::vector<TreeNode> plan(const Bsp& machine, std::span<const TreeGroup> groups,
                           const NodeBuffers& values, bool reduce) {
    if (groups.empty() || values.size() != machine.nodes())
        throw std::invalid_argument("tree groups must be non-empty and values must cover all nodes");
    std::vector<TreeNode> nodes(machine.nodes());
    for (const auto& group : groups) {
        if (group.empty()) throw std::invalid_argument("tree group must have a root");
        for (std::size_t rank = 0; rank < group.size(); ++rank) {
            const auto id = group[rank];
            if (id >= machine.nodes() || nodes[id].member)
                throw std::invalid_argument("tree node ids must be in range and groups disjoint");
            nodes[id].member = true;
            nodes[id].depth = std::bit_width(rank + 1) - 1;
            nodes[id].words = values[group[0]].size();
            if (reduce && values[id].size() != nodes[id].words)
                throw std::invalid_argument("reduction payload lengths must match within each group");
            if (rank != 0) {
                nodes[id].parent = group[(rank - 1) / 2];
                nodes[nodes[id].parent].children.push_back(id);
            }
        }
    }
    return nodes;
}
} // namespace

void tree_broadcast(Bsp& machine, Executor& executor, std::span<const TreeGroup> groups,
                    NodeBuffers& values, std::size_t tag) {
    const auto nodes = plan(machine, groups, values, false);
    const auto depth = std::max_element(nodes.begin(), nodes.end(), [](const auto& a, const auto& b) {
        return a.depth < b.depth;
    })->depth;
    if (depth == 0) return;
    for (std::size_t level = 0; level <= depth; ++level)
        machine.superstep("replication", executor, [&](Node& node) {
            const auto& entry = nodes[node.id()];
            if (!entry.member || entry.depth != level) return;
            auto& own = values[node.id()];
            if (level != 0) own = node.receive(entry.parent, tag, entry.words);
            for (auto child : entry.children) node.send(child, tag, own);
        });
}

void tree_reduce_sum(Bsp& machine, Executor& executor, std::span<const TreeGroup> groups,
                     NodeBuffers& values, std::size_t tag) {
    const auto nodes = plan(machine, groups, values, true);
    const auto depth = std::max_element(nodes.begin(), nodes.end(), [](const auto& a, const auto& b) {
        return a.depth < b.depth;
    })->depth;
    if (depth == 0) return;
    for (std::size_t remaining = depth + 1; remaining != 0; --remaining) {
        const auto level = remaining - 1;
        machine.superstep("reduction", executor, [&](Node& node) {
            const auto& entry = nodes[node.id()];
            if (!entry.member || entry.depth != level) return;
            auto& own = values[node.id()];
            for (auto child : entry.children) {
                const auto incoming = node.receive(child, tag, entry.words);
                for (std::size_t i = 0; i < own.size(); ++i) own[i] += incoming[i];
                node.compute(own.size());
            }
            if (level != 0) node.send(entry.parent, tag, own);
        });
    }
}

} // namespace dmm
