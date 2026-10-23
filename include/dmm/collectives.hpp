#pragma once

#include "dmm/bsp.hpp"
#include <span>
#include <vector>

namespace dmm {

// Groups are disjoint, ordered lists of node ids. Position 0 is the root;
// position r has parent (r-1)/2. Other nodes are not participants.
using TreeGroup = std::vector<std::size_t>;

class BspPlan;
// Append the same fixed trees to an unsealed plan. Every participant's buffer is
// already allocated; regions describe each group's shared matrix rectangle.
void tree_broadcast(BspPlan& plan, std::span<const TreeGroup> groups,
                    std::span<const std::size_t> buffers, std::size_t tag,
                    std::span<const ElementRegion> regions);
void tree_reduce_sum(BspPlan& plan, std::span<const TreeGroup> groups,
                     std::span<const std::size_t> buffers, std::size_t tag,
                     std::span<const ElementRegion> regions);

} // namespace dmm
