#include "dmm/plan.hpp"
#include <gtest/gtest.h>
#include <type_traits>
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
