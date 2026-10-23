#include "dmm/collectives.hpp"
#include "allocation_watch.hpp"
#include "test_counts.hpp"
#include <gtest/gtest.h>

namespace {
std::vector<dmm::PayloadPart> payloads(dmm::BspPlan& plan, std::size_t words,
                                       dmm::GlobalPosition origin = {}) {
    std::vector<dmm::PayloadPart> result;
    for (std::size_t node = 0; node < plan.nodes(); ++node)
        result.push_back({plan.matrix_view(plan.allocate_buffer(node, words), 1, words), origin});
    return result;
}
} // namespace
TEST(TreeCollectivesTest, FourNodeTreeMatchesHandCountsAndNodeOrderWithTwoWorkers) {
    dmm::BspPlan plan(7);
    const std::vector<dmm::TreeGroup> groups{{5, 2, 6, 0}};
    auto values = payloads(plan, 3, {2, 3});
    auto root = plan.buffer(values[5].view.buffer);
    root[0] = 1;
    root[1] = -2;
    root[2] = 4;
    plan.buffer(values[1].view.buffer)[0] = 99;
    dmm::add_tree_broadcast(plan, groups, values);
    dmm::add_tree_reduce_sum(plan, groups, values);
    plan.finalize(7, 7, {1, 1, 1});
    dmm::ThreadExecutor executor(2);
    allocation_test::Watch watch;
    plan.execute(executor);
    auto allocations = watch.finish();
    EXPECT_EQ(allocations, 0u);
    EXPECT_EQ(root[0], 4);
    EXPECT_EQ(root[1], -8);
    EXPECT_EQ(root[2], 16);
    EXPECT_EQ(plan.buffer(values[1].view.buffer)[0], 99);
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
    EXPECT_EQ(first[0].regions[0].row, 2u);
    EXPECT_EQ(trace.supersteps[1].messages[0].source, 2u);
    EXPECT_EQ(trace.supersteps[1].messages[0].destination, 0u);
}
TEST(TreeCollectivesTest, BundlesBothOperandsAndSharesUnevenTreeLevelsWithoutAllocations) {
    dmm::BspPlan plan(7);
    const std::vector<dmm::TreeGroup> groups{{3, 1, 6, 0}, {2, 5}, {4}};
    auto a = payloads(plan, 2), b = payloads(plan, 3, {1, 0});
    dmm::NodePayloads bundles(7);
    for (std::size_t node = 0; node < 7; ++node) {
        bundles[node] = {a[node], b[node]};
        std::fill(plan.buffer(a[node].view.buffer).begin(), plan.buffer(a[node].view.buffer).end(),
                  float(node + 1));
        std::fill(plan.buffer(b[node].view.buffer).begin(), plan.buffer(b[node].view.buffer).end(),
                  float(10 * (node + 1)));
    }
    dmm::add_tree_broadcast(plan, groups, bundles);
    dmm::add_tree_reduce_sum(plan, groups, bundles);
    dmm::add_tree_broadcast(plan, std::vector<dmm::TreeGroup>{{4}}, bundles);
    plan.finalize(7, 7, {1, 1, 1});
    dmm::ThreadExecutor executor(2);
    allocation_test::Watch watch;
    plan.execute(executor);
    auto allocations = watch.finish();
    EXPECT_EQ(allocations, 0u);
    auto count = dmm::count(plan.trace());
    EXPECT_EQ(count.supersteps, 6u);
    EXPECT_EQ(count.messages, 8u);
    EXPECT_EQ(count.total_words, 40u);
    EXPECT_EQ(count.total_flops, 20u);
    EXPECT_EQ(plan.buffer(a[3].view.buffer)[0], 16);
    EXPECT_EQ(plan.buffer(b[3].view.buffer)[0], 160);
    EXPECT_EQ(plan.buffer(a[2].view.buffer)[0], 6);
    EXPECT_EQ(plan.buffer(b[2].view.buffer)[0], 60);
    EXPECT_EQ(plan.buffer(a[4].view.buffer)[0], 5);
    for (const auto& s : plan.trace().supersteps)
        for (const auto& m : s.messages) {
            EXPECT_EQ(m.words, 5u);
            EXPECT_EQ(m.regions.size(), 2u);
        }
}
TEST(TreeCollectivesTest, RejectsInvalidGroupsOwnersShapesAndOriginsBeforeAddingSteps) {
    dmm::BspPlan plan(4);
    auto values = payloads(plan, 1);
    for (const auto& groups :
         std::vector<std::vector<dmm::TreeGroup>>{{}, {{}}, {{4}}, {{1, 1}}, {{1, 2}, {2, 3}}}) {
        EXPECT_THROW(dmm::add_tree_broadcast(plan, groups, values), std::invalid_argument);
        EXPECT_THROW(dmm::add_tree_reduce_sum(plan, groups, values), std::invalid_argument);
    }
    const std::vector<dmm::TreeGroup> groups{{0, 1}};
    auto wrong = values;
    std::swap(wrong[0], wrong[1]);
    EXPECT_THROW(dmm::add_tree_reduce_sum(plan, groups, wrong), std::invalid_argument);
    wrong = values;
    wrong[1].origin.row = 1;
    EXPECT_THROW(dmm::add_tree_broadcast(plan, groups, wrong), std::invalid_argument);
    wrong = values;
    wrong[1].view = plan.matrix_view(plan.allocate_buffer(1, 2), 1, 2);
    EXPECT_THROW(dmm::add_tree_broadcast(plan, groups, wrong), std::invalid_argument);
    wrong.pop_back();
    EXPECT_THROW(dmm::add_tree_broadcast(plan, groups, wrong), std::invalid_argument);
    EXPECT_THROW(plan.finalize(4, 4, {1, 1, 1}), std::invalid_argument);
    dmm::add_tree_broadcast(plan, groups, values);
    EXPECT_NO_THROW(plan.finalize(4, 4, {1, 1, 1}));
}
