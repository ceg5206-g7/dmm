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

class PreparedBsp;
// Append the same fixed trees to an unsealed plan. Every participant's buffer is
// already allocated; regions describe each group's shared matrix rectangle.
void tree_broadcast(PreparedBsp& plan, std::span<const TreeGroup> groups,
                    std::span<const std::size_t> buffers, std::size_t tag,
                    std::span<const ElementRegion> regions);
void tree_reduce_sum(PreparedBsp& plan, std::span<const TreeGroup> groups,
                     std::span<const std::size_t> buffers, std::size_t tag,
                     std::span<const ElementRegion> regions);

} // namespace dmm
