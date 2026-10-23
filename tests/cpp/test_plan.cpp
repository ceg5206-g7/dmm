#include "dmm/plan.hpp"
#include "test_counts.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <limits>
#include <type_traits>

namespace {
dmm::MatrixView allocate(dmm::BspPlan& p, std::size_t node, std::size_t rows, std::size_t cols) {
    return p.matrix_view(p.allocate_buffer(node, rows * cols), rows, cols);
}
class PlanTest : public ::testing::TestWithParam<std::size_t> {};
TEST_P(PlanTest, DeliversCopiedPayloadsInNextStepEvenWhenOriginalIsOverwritten) {
    auto executor = dmm::make_executor(GetParam());
    dmm::BspPlan plan(3);
    std::vector<dmm::MatrixView> owned, received;
    for (std::size_t node = 0; node < 3; ++node) {
        owned.push_back(allocate(plan, node, 1, node + 1));
        received.push_back(allocate(plan, node, 1, (node + 2) % 3 + 1));
        std::fill(plan.buffer(owned.back().buffer).begin(), plan.buffer(owned.back().buffer).end(),
                  float(node + 1));
    }
    auto send = plan.add_superstep(dmm::Phase::Replication);
    for (std::size_t node = 0; node < 3; ++node)
        send.add_send((node + 1) % 3, owned[node]);
    auto receive = plan.add_superstep(dmm::Phase::Reduction);
    for (std::size_t node = 0; node < 3; ++node) {
        receive.add_zero(owned[node]);
        receive.add_receive((node + 2) % 3, received[node]);
    }
    plan.finalize(3, 3, {1, 1, 1});
    EXPECT_THROW(plan.trace(), std::logic_error);
    plan.execute(*executor);
    for (std::size_t node = 0; node < 3; ++node) {
        for (auto v : plan.buffer(owned[node].buffer))
            EXPECT_EQ(v, 0);
        for (auto v : plan.buffer(received[node].buffer))
            EXPECT_EQ(v, float((node + 2) % 3 + 1));
        const auto& m = plan.trace().supersteps[0].messages[node];
        EXPECT_EQ(m.source, node);
        EXPECT_EQ(m.destination, (node + 1) % 3);
        EXPECT_EQ(m.words, node + 1);
        ASSERT_EQ(m.regions.size(), 1u);
        EXPECT_EQ(m.regions[0].cols, node + 1);
    }
    expect_work_and_communication(dmm::count(plan.trace()), (dmm::Counts{2, 0, 0, 6, 3, 3}));
}
TEST_P(PlanTest, CountsManyToOneTrafficAtReceiver) {
    auto executor = dmm::make_executor(GetParam());
    dmm::BspPlan plan(4);
    auto output = allocate(plan, 0, 3, 5);
    std::vector<dmm::MatrixView> data;
    for (std::size_t node = 1; node < 4; ++node)
        data.push_back(allocate(plan, node, 1, 5));
    auto send = plan.add_superstep(dmm::Phase::Replication);
    for (std::size_t node = 1; node < 4; ++node)
        send.add_send(0, data[node - 1]);
    auto receive = plan.add_superstep(dmm::Phase::Reduction);
    for (std::size_t node = 1; node < 4; ++node)
        receive.add_receive(node, output.submatrix(node - 1, 0, 1, 5));
    plan.finalize(5, 5, {1, 1, 1});
    plan.execute(*executor);
    expect_work_and_communication(dmm::count(plan.trace()), (dmm::Counts{2, 0, 0, 15, 15, 3}));
    EXPECT_EQ(plan.trace().peak_words[0], 30u);
}
TEST_P(PlanTest, BundlesOrderedStridedComponentsIntoOneMessageAndSumsOnReceive) {
    auto executor = dmm::make_executor(GetParam());
    dmm::BspPlan plan(2);
    auto a = allocate(plan, 0, 3, 4), b = allocate(plan, 0, 1, 3);
    auto x = allocate(plan, 1, 3, 4), y = allocate(plan, 1, 1, 3);
    for (std::size_t i = 0; i < 12; ++i)
        plan.buffer(a.buffer)[i] = float(i + 1);
    plan.buffer(b.buffer)[0] = 20;
    plan.buffer(b.buffer)[1] = 21;
    plan.buffer(b.buffer)[2] = 22;
    std::fill(plan.buffer(x.buffer).begin(), plan.buffer(x.buffer).end(), -1);
    std::fill(plan.buffer(y.buffer).begin(), plan.buffer(y.buffer).end(), 10);
    auto aa = a.submatrix(0, 1, 2, 2), bb = b.submatrix(0, 0, 1, 2);
    auto xx = x.submatrix(1, 1, 2, 2), yy = y.submatrix(0, 1, 1, 2);
    auto send = plan.add_superstep(dmm::Phase::Replication);
    send.add_send(1, {{aa, {1, 2}}, {bb, {3, 0}}});
    auto receive = plan.add_superstep(dmm::Phase::Reduction);
    receive.add_zero(a);
    receive.add_zero(b);
    receive.add_receive_sum(0, {xx, yy});
    plan.finalize(4, 4, {1, 1, 1});
    plan.execute(*executor);
    EXPECT_EQ(plan.buffer(x.buffer)[5], 1);
    EXPECT_EQ(plan.buffer(x.buffer)[6], 2);
    EXPECT_EQ(plan.buffer(x.buffer)[9], 5);
    EXPECT_EQ(plan.buffer(x.buffer)[10], 6);
    EXPECT_EQ(plan.buffer(x.buffer)[0], -1);
    EXPECT_EQ(plan.buffer(y.buffer)[0], 10);
    EXPECT_EQ(plan.buffer(y.buffer)[1], 30);
    EXPECT_EQ(plan.buffer(y.buffer)[2], 31);
    const auto& messages = plan.trace().supersteps[0].messages;
    ASSERT_EQ(messages.size(), 1u);
    EXPECT_EQ(messages[0].words, 6u);
    ASSERT_EQ(messages[0].regions.size(), 2u);
    EXPECT_EQ(messages[0].regions[0].row, 1u);
    EXPECT_EQ(messages[0].regions[1].row, 3u);
    expect_work_and_communication(dmm::count(plan.trace()), (dmm::Counts{2, 6, 6, 6, 6, 1}));
}
INSTANTIATE_TEST_SUITE_P(Workers, PlanTest, ::testing::Values(0, 1, 2, 5),
                         ::testing::PrintToStringParamName());
} // namespace
TEST(PlanValidationTest, ChecksTypedPlanScopedBuffersViewBoundsShapesAndAliasing) {
    static_assert(!std::is_convertible_v<std::size_t, dmm::BufferId>);
    static_assert(!std::is_move_constructible_v<dmm::BspPlan>);
    EXPECT_THROW(dmm::BspPlan{0}, std::invalid_argument);
    dmm::BspPlan plan(2), other(2);
    auto a = allocate(plan, 0, 2, 2), b = allocate(plan, 1, 2, 2);
    auto foreign = allocate(other, 0, 2, 2);
    auto step = plan.add_superstep(dmm::Phase::Multiply);
    EXPECT_THROW(step.add_zero(foreign), std::invalid_argument);
    EXPECT_THROW(plan.buffer({}), std::invalid_argument);
    EXPECT_THROW(plan.matrix_view(a.buffer, 2, 3), std::invalid_argument);
    EXPECT_THROW(plan.matrix_view(a.buffer, 1, 1, 5), std::invalid_argument);
    EXPECT_THROW(a.submatrix(2, 0, 1, 1), std::invalid_argument);
    EXPECT_THROW(step.add_copy(a, b), std::invalid_argument);
    EXPECT_THROW(step.add_copy(a, a), std::invalid_argument);
    EXPECT_THROW(step.add_gemm(a, a, a, {0, 0, 0}), std::invalid_argument);
    EXPECT_THROW(step.add_send(0, a), std::invalid_argument);
    EXPECT_THROW(step.add_send(2, a), std::invalid_argument);
    EXPECT_THROW(step.add_send(1, {{a, {}}, {b, {}}}), std::invalid_argument);
    EXPECT_THROW(step.add_send(1, std::span<const dmm::PayloadPart>{}), std::invalid_argument);
    EXPECT_THROW(step.add_receive(1, a), std::invalid_argument);
}
TEST(PlanValidationTest, EnforcesLatestStepAndPerNodeOperationOrder) {
    dmm::BspPlan plan(2);
    auto a = allocate(plan, 0, 1, 1), b = allocate(plan, 0, 1, 1), c = allocate(plan, 0, 1, 1);
    auto x = allocate(plan, 1, 1, 1);
    auto s = plan.add_superstep(dmm::Phase::Multiply);
    s.add_gemm(a, b, c, {0, 0, 0});
    EXPECT_THROW(s.add_zero(c), std::logic_error);
    s.add_send(1, c);
    EXPECT_THROW(s.add_gemm(a, b, c, {0, 0, 0}), std::logic_error);
    // Another node still accepts copies, independently of node 0's stage.
    EXPECT_NO_THROW(s.add_zero(x));
    auto r = plan.add_superstep(dmm::Phase::Reduction);
    EXPECT_THROW(s.add_send(1, a), std::logic_error);
    r.add_receive(0, x);
    plan.finalize(1, 1, {1, 1, 1});
    EXPECT_THROW(r.add_zero(x), std::logic_error);
    EXPECT_THROW(plan.allocate_buffer(0, 1), std::logic_error);
    EXPECT_THROW(plan.add_superstep(dmm::Phase::Multiply), std::logic_error);
}
TEST(PlanValidationTest, RejectsDuplicatePairsAndUnconsumedMessagesButCanFinishBuilding) {
    dmm::BspPlan plan(2);
    auto a = allocate(plan, 0, 2, 2), b = allocate(plan, 1, 2, 2);
    auto s = plan.add_superstep(dmm::Phase::Replication);
    s.add_send(1, a);
    EXPECT_THROW(s.add_send(1, a), std::invalid_argument);
    EXPECT_THROW(plan.finalize(2, 2, {1, 1, 1}), std::invalid_argument);
    auto r = plan.add_superstep(dmm::Phase::Reduction);
    EXPECT_THROW(r.add_receive(1, b), std::invalid_argument);
    EXPECT_THROW(r.add_receive(0, {b, b}), std::invalid_argument);
    r.add_receive(0, b);
    EXPECT_THROW(r.add_receive(0, b), std::invalid_argument);
    EXPECT_NO_THROW(plan.finalize(2, 2, {1, 1, 1}));
}
TEST(PlanValidationTest, MatchingRequiresAdjacentStepsAndOrderedComponentShapes) {
    for (int failure = 0; failure < 5; ++failure) {
        dmm::BspPlan plan(3);
        auto a = allocate(plan, 0, 1, 4), b = allocate(plan, 0, 2, 2);
        auto x = allocate(plan, 1, 1, 4), y = allocate(plan, 1, 2, 2);
        auto s = plan.add_superstep(dmm::Phase::Replication);
        s.add_send(1, {{a, {}}, {b, {}}});
        if (failure == 0)
            plan.add_superstep(dmm::Phase::Replication);
        auto r = plan.add_superstep(dmm::Phase::Reduction);
        if (failure == 1)
            r.add_receive(2, {x, y});
        else if (failure == 2)
            r.add_receive(0, x);
        else if (failure == 3)
            r.add_receive(0, {y, x});
        else if (failure == 4)
            r.add_receive(0, {x.submatrix(0, 0, 1, 2), y});
        else
            r.add_receive(0, {x, y});
        EXPECT_THROW(plan.finalize(4, 4, {1, 1, 1}), std::invalid_argument) << failure;
    }
}
TEST(PlanValidationTest, RejectsBadMetadataAndUnknownPhases) {
    dmm::BspPlan plan(1);
    EXPECT_THROW(plan.add_superstep(static_cast<dmm::Phase>(99)), std::invalid_argument);
    plan.add_superstep(dmm::Phase::Multiply);
    EXPECT_THROW(plan.finalize(2, 2, {0, 1, 1}), std::invalid_argument);
    EXPECT_THROW(plan.finalize(2, 2, {3, 1, 1}), std::invalid_argument);
    EXPECT_THROW(plan.finalize(3, 2, {1, 1, 1}), std::invalid_argument);
    plan.finalize(2, 2, {1, 1, 1});
    dmm::BspPlan invalid(2);
    auto a = allocate(invalid, 0, 1, 2), b = allocate(invalid, 1, 1, 2);
    auto s = invalid.add_superstep(dmm::Phase::Replication);
    s.add_send(1, dmm::PayloadPart{a, {std::numeric_limits<std::size_t>::max(), 0}});
    auto r = invalid.add_superstep(dmm::Phase::Reduction);
    r.add_receive(0, b);
    EXPECT_THROW(invalid.finalize(2, 2, {1, 1, 1}), std::invalid_argument);
}
TEST(PlanTest, StridedLocalCopyTouchesOnlyTheRectangle) {
    dmm::BspPlan plan(1);
    auto a = allocate(plan, 0, 3, 4), b = allocate(plan, 0, 3, 4);
    for (std::size_t i = 0; i < 12; ++i)
        plan.buffer(a.buffer)[i] = float(i);
    std::fill(plan.buffer(b.buffer).begin(), plan.buffer(b.buffer).end(), -1);
    auto s = plan.add_superstep(dmm::Phase::InputConversion);
    s.add_copy(a.submatrix(0, 1, 2, 2), b.submatrix(1, 2, 2, 2));
    plan.finalize(4, 4, {1, 1, 1});
    dmm::SerialExecutor e;
    plan.execute(e);
    EXPECT_EQ(plan.buffer(b.buffer)[6], 1);
    EXPECT_EQ(plan.buffer(b.buffer)[7], 2);
    EXPECT_EQ(plan.buffer(b.buffer)[10], 5);
    EXPECT_EQ(plan.buffer(b.buffer)[11], 6);
    EXPECT_EQ(plan.buffer(b.buffer)[0], -1);
}
