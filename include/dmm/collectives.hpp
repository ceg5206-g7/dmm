#pragma once

#include "dmm/bsp.hpp"
#include <span>
#include <vector>

namespace dmm {

// Groups are disjoint, ordered lists of node ids. Position 0 is the root;
// position r has parent (r-1)/2. Other nodes are not participants.
using TreeGroup = std::vector<std::size_t>;

// Only a node's own values are accessed by its task. All groups share each
// tree level's superstep. Singleton groups need no messages or supersteps.
void tree_broadcast(Bsp& machine, Executor& executor, std::span<const TreeGroup> groups,
                    NodeBuffers& values, std::size_t tag,
                    std::span<const ElementRegion> regions = {});
void tree_reduce_sum(Bsp& machine, Executor& executor, std::span<const TreeGroup> groups,
                     NodeBuffers& values, std::size_t tag,
                    std::span<const ElementRegion> regions = {});

} // namespace dmm
