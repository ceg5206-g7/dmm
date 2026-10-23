#include "dmm/bsp_plan.hpp"
#include <gtest/gtest.h>
#include <type_traits>

TEST(NodeMemoryTest, RetainedCacheCapacityAndIncomingCopiesAreBothCountedOnEveryRun) {
    static_assert(!std::is_copy_constructible_v<dmm::Buffer>);
    dmm::BspPlan plan(2);
    auto a = plan.allocate(0, 6), b = plan.allocate(1, 6);
    for (std::size_t round = 0; round < 2; ++round) {
        auto s = plan.step("replication"), r = plan.step("reduction");
        auto m = plan.send(s, 0, 1, 0, {a, 0, 6}, {0, 0, 1, 6});
        plan.receive(r, 1, m, {b, 0, 6});
    }
    plan.seal(6, 6, {1, 1, 1});
    dmm::ThreadExecutor executor(2);
    for (int repeat = 0; repeat < 2; ++repeat) {
        plan.execute(executor);
        EXPECT_EQ(plan.trace().peak_words, (std::vector<std::size_t>{12, 12}));
        EXPECT_EQ(dmm::count(plan.trace()).peak_words, 12u);
        EXPECT_EQ(dmm::count(plan.trace()).total_words, 12u);
    }
    EXPECT_THROW(plan.allocate(0, 2), std::logic_error);
}
