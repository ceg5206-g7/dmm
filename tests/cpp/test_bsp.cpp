#include "dmm/bsp_plan.hpp"
#include <gtest/gtest.h>
#include "test_counts.hpp"
#include <limits>

namespace {
class BspTest : public ::testing::TestWithParam<std::size_t> {};

TEST_P(BspTest, DeliversCopiedPayloadsInTheNextSuperstepAndRecordsThem) {
    auto executor = dmm::make_executor(GetParam());
    dmm::BspPlan plan(3);
    std::vector<std::size_t> owned, received;
    for (std::size_t node = 0; node < 3; ++node) {
        owned.push_back(plan.allocate(node, node + 1));
        received.push_back(plan.allocate(node, (node + 2) % 3 + 1));
        std::fill(plan.buffer(owned.back()).begin(), plan.buffer(owned.back()).end(), float(node + 1));
    }
    const auto send = plan.step("replication"), receive = plan.step("reduction");
    for (std::size_t node = 0; node < 3; ++node) {
        const auto next = (node + 1) % 3;
        auto message = plan.send(send, node, next, 7, {owned[node], 0, node + 1}, {0, 0, 1, node + 1});
        plan.receive(receive, next, message, {received[next], 0, node + 1});
        plan.zero(receive, node, owned[node]);
    }
    plan.seal(3, 3, {1, 1, 1});
    EXPECT_THROW(plan.trace(), std::logic_error);
    plan.execute(*executor);
    for (std::size_t node = 0; node < 3; ++node) {
        for (auto value : plan.buffer(owned[node]))
            EXPECT_EQ(value, 0);
        for (auto value : plan.buffer(received[node]))
            EXPECT_EQ(value, float((node + 2) % 3 + 1));
    }
    const auto& trace = plan.trace();
    ASSERT_EQ(trace.nodes, 3u);
    ASSERT_EQ(trace.supersteps.size(), 2u);
    for (std::size_t node = 0; node < 3; ++node) {
        const auto& m = trace.supersteps[0].messages[node];
        EXPECT_EQ(m.source, node);
        EXPECT_EQ(m.destination, (node + 1) % 3);
        EXPECT_EQ(m.tag, 7u);
        EXPECT_EQ(m.words, node + 1);
    }
    expect_work_and_communication(dmm::count(trace), (dmm::Counts{2, 0, 0, 6, 3, 3}));
    expect_work_and_communication(dmm::count(trace.supersteps[1]), (dmm::Counts{1, 0, 0, 0, 0, 0}));
}

TEST_P(BspTest, CountsManyToOneTrafficAtTheReceiver) {
    auto executor = dmm::make_executor(GetParam());
    dmm::BspPlan plan(4);
    auto output = plan.allocate(0, 15);
    const auto send = plan.step("replication"), receive = plan.step("reduction");
    for (std::size_t node = 1; node < 4; ++node) {
        auto data = plan.allocate(node, 5);
        auto message = plan.send(send, node, 0, 0, {data, 0, 5}, {0, 0, 1, 5});
        plan.receive(receive, 0, message, {output, (node - 1) * 5, 5});
    }
    plan.seal(5, 5, {1, 1, 1});
    plan.execute(*executor);
    expect_work_and_communication(dmm::count(plan.trace()), (dmm::Counts{2, 0, 0, 15, 15, 3}));
    EXPECT_EQ(plan.trace().peak_words[0], 30u);
}
INSTANTIATE_TEST_SUITE_P(Workers, BspTest, ::testing::Values(0, 1, 2, 5),
                         ::testing::PrintToStringParamName());

TEST(BspValidationTest, RejectsInvalidViewsDestinationsAndUnmatchedReceivesBeforeExecution) {
    EXPECT_THROW(dmm::BspPlan{0}, std::invalid_argument);
    dmm::BspPlan plan(2);
    const auto a = plan.allocate(0, 2), b = plan.allocate(1, 2);
    const auto s = plan.step("replication");
    EXPECT_THROW(plan.send(s, 0, 2, 0, {a, 0, 2}, {0, 0, 1, 2}), std::invalid_argument);
    EXPECT_THROW(plan.send(s, 0, 0, 0, {a, 0, 2}, {0, 0, 1, 2}), std::invalid_argument);
    EXPECT_THROW(plan.send(s, 0, 1, 0, {b, 0, 2}, {0, 0, 1, 2}), std::invalid_argument);
    EXPECT_THROW(plan.send(s, 0, 1, 0, {a, 0, 2}, {0, 0, 2, 2}), std::invalid_argument);
    auto m = plan.send(s, 0, 1, 0, {a, 0, 2}, {0, 0, 1, 2});
    EXPECT_THROW(plan.send(s, 0, 1, 0, {a, 0, 2}, {0, 0, 1, 2}), std::invalid_argument);
    EXPECT_THROW(plan.receive(s, 1, m, {b, 0, 2}), std::invalid_argument);
    EXPECT_THROW(plan.seal(2, 2, {1, 1, 1}), std::invalid_argument);
    const auto r = plan.step("reduction");
    EXPECT_THROW(plan.receive(r, 0, m, {a, 0, 2}), std::invalid_argument);
    EXPECT_THROW(plan.receive(r, 1, m, {b, 1, 2}), std::invalid_argument);
    plan.receive(r, 1, m, {b, 0, 2});
    EXPECT_THROW(plan.receive(r, 1, m, {b, 0, 2}), std::invalid_argument);
    plan.seal(2, 2, {1, 1, 1});
    EXPECT_THROW(plan.allocate(0, 2), std::logic_error);
}

TEST(BspValidationTest, RejectsOutOfMatrixRegionsAtSeal) {
    dmm::BspPlan plan(2);
    auto a = plan.allocate(0, 2), b = plan.allocate(1, 2);
    auto s = plan.step("replication"), r = plan.step("reduction");
    auto m = plan.send(s, 0, 1, 0, {a, 0, 2}, {std::numeric_limits<std::size_t>::max(), 0, 1, 2});
    plan.receive(r, 1, m, {b, 0, 2});
    EXPECT_THROW(plan.seal(2, 2, {1, 1, 1}), std::invalid_argument);
}

TEST(BspValidationTest, RejectsInvalidCoverageAndUnknownPhasesBeforeExecution) {
    dmm::BspPlan plan(1);
    EXPECT_THROW(plan.step("unknown"), std::invalid_argument);
    plan.step("multiply");
    EXPECT_THROW(plan.seal(2, 2, {0, 1, 1}), std::invalid_argument);
    EXPECT_THROW(plan.seal(2, 2, {3, 1, 1}), std::invalid_argument);
    plan.seal(2, 2, {1, 1, 1});
}

TEST(BspCostTest, ChargesCriticalWorkWordsAndSupersteps) {
    const dmm::Counts counts{5, 128, 8, 120, 8, 120};
    EXPECT_DOUBLE_EQ(dmm::bsp_cost(counts, {}), 8);
    EXPECT_DOUBLE_EQ(dmm::bsp_cost(counts, {1, 10, 100}), 588);
    EXPECT_DOUBLE_EQ(dmm::bsp_cost(counts, {.5, 0, 0}), 4);
    EXPECT_THROW(dmm::bsp_cost(counts, {1, -1, 0}), std::invalid_argument);
    EXPECT_THROW(dmm::bsp_cost(counts, {1, 0, std::numeric_limits<double>::infinity()}),
                 std::invalid_argument);
}
} // namespace
