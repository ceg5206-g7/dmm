#include "dmm/plan.hpp"
#include "test_counts.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <string>
#include <type_traits>
#include <vector>
namespace {
dmm::MatrixView allocate(dmm::BspPlan& plan, std::size_t node, std::size_t words) {
    return plan.matrix_view(plan.allocate_buffer(node, words), 1, words);
}
} // namespace
TEST(NodeMemoryTest, RetainedCacheCapacityAndIncomingCopiesAreCountedOnEveryRun) {
    static_assert(!std::is_copy_constructible_v<dmm::Buffer>);
    dmm::BspPlan plan(2);
    auto a = plan.matrix_view(plan.allocate_buffer(0, 6), 1, 6),
         b = plan.matrix_view(plan.allocate_buffer(1, 6), 1, 6);
    for (int round = 0; round < 2; ++round) {
        auto s = plan.add_superstep(dmm::Phase::Replication);
        s.add_send(1, a);
        auto r = plan.add_superstep(dmm::Phase::Reduction);
        r.add_receive(0, b);
    }
    plan.finalize(6, 6, {1, 1, 1});
    dmm::ThreadExecutor executor(2);
    for (int repeat = 0; repeat < 2; ++repeat) {
        plan.execute(executor);
        EXPECT_EQ(plan.trace().peak_words, (std::vector<std::size_t>{12, 12}));
        EXPECT_EQ(dmm::count(plan.trace()).peak_words, 12u);
        EXPECT_EQ(dmm::count(plan.trace()).total_words, 12u);
    }
}
TEST(DeliveryMergeTest, FoldsDeliveryOnlyStepsIntoTheirSuccessor) {
    for (const bool merge : {false, true})
        for (const bool sum : {false, true}) {
            SCOPED_TRACE(::testing::Message() << "merge=" << merge << " sum=" << sum);
            dmm::BspPlan plan(2);
            auto a = allocate(plan, 0, 2), b = allocate(plan, 1, 2), d = allocate(plan, 1, 2),
                 e = allocate(plan, 0, 2);
            plan.add_superstep(dmm::Phase::Skew).add_send(1, a);
            auto deliver = plan.add_superstep(dmm::Phase::InputConversion);
            sum ? deliver.add_receive_sum(0, b) : deliver.add_receive(0, b);
            auto work = plan.add_superstep(dmm::Phase::Multiply);
            work.add_copy(b, d);
            work.add_send(0, d);
            plan.add_superstep(dmm::Phase::OutputConversion).add_receive(1, e);
            plan.finalize(2, 2, {1, 1, 1}, merge);
            auto input = plan.buffer(a.buffer);
            input[0] = 2;
            input[1] = 5;
            std::fill(plan.buffer(b.buffer).begin(), plan.buffer(b.buffer).end(), 0.f);
            dmm::SerialExecutor executor;
            plan.execute(executor);
            const auto& trace = plan.trace();
            // A receive-sum computes, so it keeps its own superstep.
            const bool merged = merge && !sum;
            expect_work_and_communication(dmm::count(trace),
                                          (dmm::Counts{merged ? 3u : 4u, sum ? 2u : 0u, sum ? 2u : 0u, 4, 4, 2}));
            std::vector<std::string> labels;
            for (const auto& step : trace.supersteps)
                labels.push_back(step.label);
            EXPECT_EQ(labels, merged ? (std::vector<std::string>{"skew", "multiply", "output_conversion"})
                                     : (std::vector<std::string>{"skew", "input_conversion", "multiply",
                                                                 "output_conversion"}));
            EXPECT_EQ(plan.buffer(e.buffer)[0], 2.f);
            EXPECT_EQ(plan.buffer(e.buffer)[1], 5.f);
        }
}
TEST(DeliveryMergeTest, KeepsMalformedSchedulesRejected) {
    dmm::BspPlan plan(2);
    auto a = allocate(plan, 0, 2), b = allocate(plan, 1, 2), c = allocate(plan, 1, 2);
    plan.add_superstep(dmm::Phase::Skew).add_send(1, a);
    plan.add_superstep(dmm::Phase::Multiply).add_receive(0, b);
    // Nothing was sent in step 1, so this receive must not be matched to step 0.
    plan.add_superstep(dmm::Phase::Multiply).add_receive(0, c);
    EXPECT_THROW(plan.finalize(2, 2, {1, 1, 1}, true), std::invalid_argument);
}
