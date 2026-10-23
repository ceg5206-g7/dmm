#include "dmm/collectives.hpp"
#include <bit>

namespace dmm {
namespace {
struct TreeNode {
    bool member = false;
    std::size_t depth = 0, parent = 0;
    std::vector<std::size_t> children;
};
void append_tree(BspPlan& plan, std::span<const TreeGroup> groups, const NodePayloads& payloads,
                 bool reduce) {
    if (groups.empty() || payloads.size() != plan.nodes())
        throw std::invalid_argument("tree needs groups and one payload list per node");
    std::vector<TreeNode> nodes(plan.nodes());
    std::vector<std::vector<MatrixView>> targets(plan.nodes());
    std::size_t depth = 0;
    for (const auto& group : groups) {
        if (group.empty() || group[0] >= plan.nodes())
            throw std::invalid_argument("tree needs a valid root");
        const auto& root = payloads[group[0]];
        if (root.empty())
            throw std::invalid_argument("tree payload must be nonempty");
        for (std::size_t rank = 0; rank < group.size(); ++rank) {
            const auto node = group[rank];
            if (node >= plan.nodes() || nodes[node].member || payloads[node].size() != root.size())
                throw std::invalid_argument("tree groups must be disjoint with matching component counts");
            auto& entry = nodes[node];
            entry.member = true;
            entry.depth = std::bit_width(rank + 1) - 1;
            depth = std::max(depth, entry.depth);
            for (std::size_t p = 0; p < root.size(); ++p) {
                const auto& part = payloads[node][p];
                const auto& reference = root[p];
                if (plan.owner(part.view) != node || part.view.rows != reference.view.rows ||
                    part.view.cols != reference.view.cols || part.origin.row != reference.origin.row ||
                    part.origin.col != reference.origin.col)
                    throw std::invalid_argument(
                        "tree components need matching shapes/regions and correct owners");
                targets[node].push_back(part.view);
            }
            if (rank) {
                entry.parent = group[(rank - 1) / 2];
                nodes[entry.parent].children.push_back(node);
            }
        }
    }
    if (!depth)
        return;
    for (std::size_t round = 0; round <= depth; ++round) {
        const auto level = reduce ? depth - round : round;
        auto step = plan.add_superstep(reduce ? Phase::Reduction : Phase::Replication);
        for (std::size_t node = 0; node < nodes.size(); ++node) {
            const auto& entry = nodes[node];
            if (!entry.member || entry.depth != level)
                continue;
            if (reduce) {
                for (auto child : entry.children)
                    step.add_receive_sum(child, targets[node]);
                if (level)
                    step.add_send(entry.parent, payloads[node]);
            } else {
                if (level)
                    step.add_receive(entry.parent, targets[node]);
                for (auto child : entry.children)
                    step.add_send(child, payloads[node]);
            }
        }
    }
}
NodePayloads single_components(std::span<const PayloadPart> payloads) {
    NodePayloads result(payloads.size());
    for (std::size_t node = 0; node < payloads.size(); ++node)
        if (payloads[node].view.rows)
            result[node].push_back(payloads[node]);
    return result;
}
} // namespace
void add_tree_broadcast(BspPlan& plan, std::span<const TreeGroup> groups, const NodePayloads& payloads) {
    append_tree(plan, groups, payloads, false);
}
void add_tree_reduce_sum(BspPlan& plan, std::span<const TreeGroup> groups, const NodePayloads& payloads) {
    append_tree(plan, groups, payloads, true);
}
void add_tree_broadcast(BspPlan& plan, std::span<const TreeGroup> groups,
                        std::span<const PayloadPart> payloads) {
    append_tree(plan, groups, single_components(payloads), false);
}
void add_tree_reduce_sum(BspPlan& plan, std::span<const TreeGroup> groups,
                         std::span<const PayloadPart> payloads) {
    append_tree(plan, groups, single_components(payloads), true);
}
} // namespace dmm
