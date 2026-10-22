#include "dmm/collectives.hpp"
#include <gtest/gtest.h>
#include "test_counts.hpp"
#include <vector>

TEST(TreeCollectivesTest, FourNodeTreeMatchesHandCountsAndNodeOrderWithTwoWorkers) {
    dmm::ThreadExecutor executor(2);
    dmm::Bsp machine(7);
    const std::vector<dmm::TreeGroup> groups{{5, 2, 6, 0}};
    auto values = machine.buffers();
    values[5].assign(std::vector<float>{1, -2, 4});
    values[1].assign(std::vector<float>{99});
    dmm::tree_broadcast(machine, executor, groups, values, 40);
    for (auto id : groups[0]) EXPECT_EQ(values[id], values[5]);
    EXPECT_EQ(values[1], (std::vector<float>{99}));
    auto trace = machine.finish();
    expect_work_and_communication(dmm::count(trace), (dmm::Counts{3, 0, 0, 9, 9, 3}));
    ASSERT_EQ(trace.supersteps[0].messages.size(), 2u);
    EXPECT_EQ(trace.supersteps[0].messages[0].source, 5u);
    EXPECT_EQ(trace.supersteps[0].messages[0].destination, 2u);
    EXPECT_EQ(trace.supersteps[0].messages[1].destination, 6u);
    EXPECT_EQ(trace.supersteps[1].messages[0].source, 2u);
    EXPECT_EQ(trace.supersteps[1].messages[0].destination, 0u);
    dmm::tree_reduce_sum(machine, executor, groups, values, 41);
    EXPECT_EQ(values[5], (std::vector<float>{4, -8, 16}));
    expect_work_and_communication(dmm::count(machine.finish()), (dmm::Counts{3, 9, 9, 9, 9, 3}));
}

TEST(TreeCollectivesTest, DisjointUnevenTreesShareLevelsAndSingletonDoesNothing) {
    dmm::SerialExecutor executor;
    dmm::Bsp machine(8);
    const std::vector<dmm::TreeGroup> groups{{0, 2, 4}, {7, 5}, {1}};
    auto values = machine.buffers();
    values[0].assign(std::vector<float>{2}); values[7].assign(std::vector<float>{3}); values[1].assign(std::vector<float>{4});
    dmm::tree_broadcast(machine, executor, groups, values, 0);
    EXPECT_EQ(values[2], values[0]); EXPECT_EQ(values[4], values[0]);
    EXPECT_EQ(values[5], values[7]);
    EXPECT_EQ(dmm::count(machine.finish()).supersteps, 2u);
    dmm::tree_reduce_sum(machine, executor, groups, values, 0);
    EXPECT_EQ(values[0], (std::vector<float>{6}));
    EXPECT_EQ(values[7], (std::vector<float>{6}));
    EXPECT_EQ(values[1], (std::vector<float>{4}));
    const auto counts = dmm::count(machine.finish());
    EXPECT_EQ(counts.supersteps, 2u); EXPECT_EQ(counts.total_flops, 3u);
    const std::vector<dmm::TreeGroup> singleton{{1}};
    dmm::tree_broadcast(machine, executor, singleton, values, 0);
    dmm::tree_reduce_sum(machine, executor, singleton, values, 0);
    EXPECT_EQ(dmm::count(machine.finish()).supersteps, 0u);
}

TEST(TreeCollectivesTest, RejectsInvalidParticipantsAndLengthsBeforeAnySuperstep) {
    dmm::SerialExecutor executor;
    dmm::Bsp machine(4);
    auto values = machine.buffers(1);
    for (auto& value : values) value[0] = 1;
    for (const auto& groups : std::vector<std::vector<dmm::TreeGroup>>{
            {}, {{}}, {{4}}, {{1, 1}}, {{1, 2}, {2, 3}}}) {
        EXPECT_THROW(dmm::tree_broadcast(machine, executor, groups, values, 0), std::invalid_argument);
        EXPECT_THROW(dmm::tree_reduce_sum(machine, executor, groups, values, 0), std::invalid_argument);
        EXPECT_TRUE(machine.finish().supersteps.empty());
    }
    values[1].clear();
    const std::vector<dmm::TreeGroup> groups{{0, 1}};
    EXPECT_THROW(dmm::tree_reduce_sum(machine, executor, groups, values, 0), std::invalid_argument);
    values.pop_back();
    EXPECT_THROW(dmm::tree_broadcast(machine, executor, groups, values, 0), std::invalid_argument);
}
