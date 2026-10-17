#include "dmm/execution.hpp"
#include <gtest/gtest.h>
#include <chrono>
#include <cstddef>
#include <future>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
TEST(ExecutorLifecycleTest, RejectsZeroThreadCount) {
    EXPECT_THROW(dmm::ThreadExecutor{0}, std::invalid_argument);
}

TEST(ExecutorLifecycleTest, ClosesAndRejectsReuse) {
    dmm::ThreadExecutor executor(2);
    EXPECT_EQ(&executor.enter(), &executor);
    EXPECT_THROW(executor.enter(), std::runtime_error);
    EXPECT_NO_THROW(executor.for_each(1, [](std::size_t) {}));
    EXPECT_NO_THROW(executor.close());
    EXPECT_NO_THROW(executor.close());
    EXPECT_THROW(executor.enter(), std::runtime_error);
    EXPECT_THROW(executor.for_each(1, [](std::size_t) {}), std::runtime_error);
}

TEST(ExecutorLifecycleTest, CloseWaitsForActiveBatch) {
    dmm::ThreadExecutor executor(2);
    std::promise<void> started, release;
    auto running = started.get_future();
    auto proceed = release.get_future();
    auto batch = std::async(std::launch::async, [&] {
        executor.for_each(1, [&](std::size_t) {
            started.set_value();
            proceed.wait();
        });
    });
    running.wait();
    auto closing = std::async(std::launch::async, [&] { executor.close(); });
    const auto status = closing.wait_for(std::chrono::milliseconds(0));
    release.set_value();
    EXPECT_NO_THROW(batch.get());
    EXPECT_NO_THROW(closing.get());
    EXPECT_EQ(status, std::future_status::timeout);
}

class ExecutorTest : public ::testing::TestWithParam<std::size_t> {};

TEST_P(ExecutorTest, PropagatesErrorsAndReusesWorkers) {
    const auto workers = GetParam();
    auto executor = dmm::make_executor(workers);
    EXPECT_THROW(executor->for_each(7, [](std::size_t i) {
        if (i == 2) throw std::runtime_error("worker failure");
    }), std::runtime_error);
    std::vector<int> counts(17);
    std::vector<std::thread::id> threads(counts.size());
    executor->for_each(counts.size(), [&](std::size_t i) {
        ++counts[i];
        threads[i] = std::this_thread::get_id();
    });
    const auto caller = std::this_thread::get_id();
    for (std::size_t i = 0; i < counts.size(); ++i) {
        SCOPED_TRACE(::testing::Message() << "task=" << i);
        EXPECT_EQ(counts[i], 1);
        EXPECT_EQ(threads[i] == caller, workers == 0);
    }
    EXPECT_NO_THROW(executor->for_each(0, [](std::size_t) {
        throw std::runtime_error("empty batch");
    }));
}

INSTANTIATE_TEST_SUITE_P(Workers, ExecutorTest, ::testing::Values(0, 1, 2, 5),
                         ::testing::PrintToStringParamName());
} // namespace
