#pragma once
#include "dmm/plan.hpp"

namespace dmm {
// Groups are disjoint ordered node lists; rank 0 is root, parent(r)=(r-1)/2.
using TreeGroup = std::vector<std::size_t>;
using NodePayloads = std::vector<std::vector<PayloadPart>>;
void add_tree_broadcast(BspPlan&, std::span<const TreeGroup>, const NodePayloads&);
void add_tree_reduce_sum(BspPlan&, std::span<const TreeGroup>, const NodePayloads&);
// One component per node convenience forms. Nonparticipants may have empty views.
void add_tree_broadcast(BspPlan&, std::span<const TreeGroup>, std::span<const PayloadPart>);
void add_tree_reduce_sum(BspPlan&, std::span<const TreeGroup>, std::span<const PayloadPart>);
} // namespace dmm
