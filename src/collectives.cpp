#include "dmm/collectives.hpp"
#include "dmm/prepared.hpp"
#include <algorithm>
#include <bit>
#include <limits>
#include <stdexcept>

namespace dmm {
namespace {
struct TreeNode {
    bool member = false;
    std::size_t depth = 0, parent = 0, words = 0;
    std::vector<std::size_t> children;
    std::optional<ElementRegion> region;
};

std::vector<TreeNode> plan(std::size_t count, std::span<const TreeGroup> groups,
                           std::span<const std::size_t> sizes, bool reduce, std::span<const ElementRegion> regions) {
    if (groups.empty() || sizes.size() != count)
        throw std::invalid_argument("tree groups must be non-empty and values must cover all nodes");
    if (!regions.empty() && regions.size() != groups.size())
        throw std::invalid_argument("tree regions must match the number of groups");
    std::vector<TreeNode> nodes(count);
    for (std::size_t g = 0; g < groups.size(); ++g) {
        const auto& group = groups[g];
        if (group.empty()) throw std::invalid_argument("tree group must have a root");
        for (std::size_t rank = 0; rank < group.size(); ++rank) {
            const auto id = group[rank];
            if (id >= count || nodes[id].member)
                throw std::invalid_argument("tree node ids must be in range and groups disjoint");
            nodes[id].member = true;
            nodes[id].depth = std::bit_width(rank + 1) - 1;
            nodes[id].words = sizes[group[0]];
            if (!regions.empty()) {
                const auto& r = regions[g];
                if (r.rows == 0 || r.cols == 0 || r.rows > std::numeric_limits<std::size_t>::max() / r.cols ||
                    r.rows * r.cols != nodes[id].words || r.row > std::numeric_limits<std::size_t>::max() - r.rows ||
                    r.col > std::numeric_limits<std::size_t>::max() - r.cols)
                    throw std::invalid_argument("tree region must match its payload without overflow");
                nodes[id].region = r;
            }
            if (reduce && sizes[id] != nodes[id].words)
                throw std::invalid_argument("reduction payload lengths must match within each group");
            if (rank != 0) {
                nodes[id].parent = group[(rank - 1) / 2];
                nodes[nodes[id].parent].children.push_back(id);
            }
        }
    }
    return nodes;
}
std::vector<std::size_t> payload_sizes(const Bsp& machine, const NodeBuffers& values) {
    if (values.size() != machine.nodes()) throw std::invalid_argument("tree values must cover all nodes");
    std::vector<std::size_t> sizes;
    for (std::size_t id = 0; id < machine.nodes(); ++id) {
        if (!machine.owns(id, values[id])) throw std::invalid_argument("tree payload belongs to a different logical node");
        sizes.push_back(values[id].size());
    }
    return sizes;
}

void prepared_tree(PreparedBsp& machine, std::span<const TreeGroup> groups,
                   std::span<const std::size_t> buffers, std::size_t tag,
                   std::span<const ElementRegion> regions, bool reduce) {
    if (buffers.size() != machine.nodes() || regions.size() != groups.size())
        throw std::invalid_argument("prepared tree needs one buffer per node and one region per group");
    std::vector<std::size_t> sizes;
    for (std::size_t id = 0; id < machine.nodes(); ++id) {
        if (!machine.owns(id, buffers[id])) throw std::invalid_argument("tree payload belongs to a different logical node");
        sizes.push_back(machine.buffer(buffers[id]).size());
    }
    const auto nodes = plan(machine.nodes(), groups, sizes, true, regions);
    const auto depth = std::max_element(nodes.begin(), nodes.end(), [](const auto& a, const auto& b) {
        return a.depth < b.depth;
    })->depth;
    if (depth == 0) return;
    std::vector<std::size_t> steps, messages(nodes.size());
    for (std::size_t level = 0; level <= depth; ++level) steps.push_back(machine.step(reduce ? "reduction" : "replication"));
    for (std::size_t round = 0; round <= depth; ++round) {
        const auto level = reduce ? depth - round : round, step = steps[round];
        for (std::size_t id = 0; id < nodes.size(); ++id) {
            const auto& entry = nodes[id];
            if (!entry.member || entry.depth != level) continue;
            const BufferView own{buffers[id], 0, entry.region->cols};
            if (reduce) {
                for (auto child : entry.children) machine.receive(step, id, messages[child], own, true);
                if (level != 0) messages[id] = machine.send(step, id, entry.parent, tag, own, *entry.region);
            } else {
                if (level != 0) machine.receive(step, id, messages[id], own);
                for (auto child : entry.children) messages[child] = machine.send(step, id, child, tag, own, *entry.region);
            }
        }
    }
}
} // namespace

void tree_broadcast(Bsp& machine, Executor& executor, std::span<const TreeGroup> groups,
                    NodeBuffers& values, std::size_t tag, std::span<const ElementRegion> regions) {
    const auto nodes = plan(machine.nodes(), groups, payload_sizes(machine, values), false, regions);
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
            for (auto child : entry.children) node.send(child, tag, own, entry.region);
        });
}

void tree_reduce_sum(Bsp& machine, Executor& executor, std::span<const TreeGroup> groups,
                     NodeBuffers& values, std::size_t tag, std::span<const ElementRegion> regions) {
    const auto nodes = plan(machine.nodes(), groups, payload_sizes(machine, values), true, regions);
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
            if (level != 0) node.send(entry.parent, tag, own, entry.region);
        });
    }
}

void tree_broadcast(PreparedBsp& machine, std::span<const TreeGroup> groups,
                    std::span<const std::size_t> buffers, std::size_t tag, std::span<const ElementRegion> regions) {
    prepared_tree(machine, groups, buffers, tag, regions, false);
}
void tree_reduce_sum(PreparedBsp& machine, std::span<const TreeGroup> groups,
                     std::span<const std::size_t> buffers, std::size_t tag, std::span<const ElementRegion> regions) {
    prepared_tree(machine, groups, buffers, tag, regions, true);
}

} // namespace dmm
