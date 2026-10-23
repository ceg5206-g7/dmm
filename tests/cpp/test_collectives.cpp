#include "dmm/collectives.hpp"
#include "dmm/bsp_plan.hpp"
#include <gtest/gtest.h>
#include "test_counts.hpp"

TEST(TreeCollectivesTest, FourNodeTreeMatchesHandCountsAndNodeOrderWithTwoWorkers) {
    dmm::BspPlan plan(7);
    const std::vector<dmm::TreeGroup> groups{{5, 2, 6, 0}};
    std::vector<std::size_t> values;
    for (std::size_t node = 0; node < 7; ++node)
        values.push_back(plan.allocate(node, 3));
    auto root = plan.buffer(values[5]);
    root[0] = 1;
    root[1] = -2;
    root[2] = 4;
    plan.buffer(values[1])[0] = 99;
    const std::vector<dmm::ElementRegion> regions{{2, 3, 1, 3}};
    dmm::tree_broadcast(plan, groups, values, 40, regions);
    dmm::tree_reduce_sum(plan, groups, values, 41, regions);
    plan.seal(7, 7, {1, 1, 1});
    dmm::ThreadExecutor executor(2);
    plan.execute(executor);
    EXPECT_EQ(plan.buffer(values[5])[0], 4);
    EXPECT_EQ(plan.buffer(values[5])[1], -8);
    EXPECT_EQ(plan.buffer(values[5])[2], 16);
    EXPECT_EQ(plan.buffer(values[1])[0], 99);
    const auto& trace = plan.trace();
    dmm::Counts broadcast, reduction;
    for (std::size_t s = 0; s < 6; ++s)
        (s < 3 ? broadcast : reduction) += dmm::count(trace.supersteps[s]);
    expect_work_and_communication(broadcast, (dmm::Counts{3, 0, 0, 9, 9, 3}));
    expect_work_and_communication(reduction, (dmm::Counts{3, 9, 9, 9, 9, 3}));
    const auto& first = trace.supersteps[0].messages;
    ASSERT_EQ(first.size(), 2u);
    EXPECT_EQ(first[0].source, 5u);
    EXPECT_EQ(first[0].destination, 2u);
    EXPECT_EQ(first[1].destination, 6u);
    EXPECT_EQ(first[0].region->row, 2u);
    EXPECT_EQ(trace.supersteps[1].messages[0].source, 2u);
    EXPECT_EQ(trace.supersteps[1].messages[0].destination, 0u);
}

TEST(TreeCollectivesTest, DisjointUnevenTreesShareLevelsAndSingletonDoesNothing) {
    dmm::BspPlan plan(8);
    const std::vector<dmm::TreeGroup> groups{{0, 2, 4}, {7, 5}, {1}};
    const std::vector<dmm::ElementRegion> regions(3, {0, 0, 1, 1});
    std::vector<std::size_t> values;
    for (std::size_t node = 0; node < 8; ++node)
        values.push_back(plan.allocate(node, 1));
    plan.buffer(values[0])[0] = 2;
    plan.buffer(values[7])[0] = 3;
    plan.buffer(values[1])[0] = 4;
    dmm::tree_broadcast(plan, groups, values, 0, regions);
    dmm::tree_reduce_sum(plan, groups, values, 0, regions);
    dmm::tree_broadcast(plan, std::vector<dmm::TreeGroup>{{1}}, values, 0,
                        std::vector<dmm::ElementRegion>{{0, 0, 1, 1}});
    plan.seal(8, 8, {1, 1, 1});
    dmm::SerialExecutor executor;
    plan.execute(executor);
    EXPECT_EQ(plan.buffer(values[0])[0], 6);
    EXPECT_EQ(plan.buffer(values[7])[0], 6);
    EXPECT_EQ(plan.buffer(values[1])[0], 4);
    EXPECT_EQ(dmm::count(plan.trace()).supersteps, 4u);
    EXPECT_EQ(dmm::count(plan.trace()).total_flops, 3u);
}

TEST(TreeCollectivesTest, RejectsInvalidParticipantsOwnersAndLengthsBeforeAddingSteps) {
    dmm::BspPlan plan(4);
    std::vector<std::size_t> values;
    for (std::size_t node = 0; node < 4; ++node)
        values.push_back(plan.allocate(node, 1));
    for (const auto& groups :
         std::vector<std::vector<dmm::TreeGroup>>{{}, {{}}, {{4}}, {{1, 1}}, {{1, 2}, {2, 3}}}) {
        std::vector<dmm::ElementRegion> regions(groups.size(), {0, 0, 1, 1});
        EXPECT_THROW(dmm::tree_broadcast(plan, groups, values, 0, regions), std::invalid_argument);
        EXPECT_THROW(dmm::tree_reduce_sum(plan, groups, values, 0, regions), std::invalid_argument);
    }
    const std::vector<dmm::TreeGroup> groups{{0, 1}};
    const std::vector<dmm::ElementRegion> right{{0, 0, 1, 1}}, wrong{{0, 0, 1, 2}};
    EXPECT_THROW(dmm::tree_broadcast(plan, groups, values, 0, wrong), std::invalid_argument);
    std::swap(values[0], values[1]);
    EXPECT_THROW(dmm::tree_reduce_sum(plan, groups, values, 0, right), std::invalid_argument);
    std::swap(values[0], values[1]);
    values.pop_back();
    EXPECT_THROW(dmm::tree_broadcast(plan, groups, values, 0, right), std::invalid_argument);
    EXPECT_THROW(plan.seal(4, 4, {1, 1, 1}), std::invalid_argument);
}
