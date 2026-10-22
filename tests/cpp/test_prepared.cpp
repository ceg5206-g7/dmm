#include "dmm/prepared.hpp"
#include "dmm/cannon.hpp"
#include "dmm/coverage.hpp"
#include <gtest/gtest.h>
#include <atomic>
#include <cstdlib>
#include <new>

namespace {
std::atomic<bool> watch_allocations{false};
std::atomic<std::size_t> allocations{0};
}
// Catch payload and metadata allocations on worker threads as well as the caller.
void* operator new(std::size_t bytes) {
    if (watch_allocations.load(std::memory_order_relaxed)) ++allocations;
    if (auto* pointer = std::malloc(bytes ? bytes : 1)) return pointer;
    throw std::bad_alloc();
}
void operator delete(void* pointer) noexcept { std::free(pointer); }
void* operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete[](void* pointer) noexcept { ::operator delete(pointer); }

TEST(PreparedTest, RepeatedPaddedProductsMatchReferenceTraceAndAllocateNothing) {
    for (auto P : {1u, 4u, 16u}) for (auto workers : {0u, 2u}) {
        auto executor = dmm::make_executor(workers);
        dmm::Matrix<float> a(17), b(17);
        for (std::size_t i = 0; i < a.values.size(); ++i) {
            a.values[i] = static_cast<float>(static_cast<int>(i % 13) - 6) / 32;
            b.values[i] = static_cast<float>(static_cast<int>(i % 7) - 3) / 16;
        }
        const auto expected = dmm::Cannon{P}.run(a, b, *executor);
        dmm::PreparedCannon plan(a, b, P);
        std::vector<float> output(a.values.size());
        EXPECT_THROW(plan.trace(), std::logic_error);
        for (int repeat = 0; repeat < 3; ++repeat) {
            allocations = 0; watch_allocations = true;
            const auto times = plan.run_into(output, *executor);
            watch_allocations = false;
            EXPECT_EQ(allocations.load(), 0u) << "P=" << P << ", H=" << workers;
            EXPECT_EQ(output, expected.product.values);
            EXPECT_GT(times.end_to_end.total, 0);
            EXPECT_NEAR(times.end_to_end.total, times.end_to_end.compute + times.end_to_end.copy +
                        times.end_to_end.synchronization, 1e-10);
            const auto trace = plan.trace();
            EXPECT_NO_THROW(dmm::validate_coverage(trace));
            const auto actual = dmm::count(trace), original = dmm::count(expected.trace);
            EXPECT_EQ(actual.total_flops, original.total_flops);
            EXPECT_EQ(actual.critical_flops, original.critical_flops);
            EXPECT_EQ(actual.total_words, original.total_words);
            EXPECT_EQ(actual.critical_words, original.critical_words);
            EXPECT_EQ(actual.supersteps, original.supersteps);
            EXPECT_EQ(actual.messages, original.messages);
            for (std::size_t s = 0; s < trace.supersteps.size(); ++s) {
                const auto& messages = trace.supersteps[s].messages;
                const auto& originals = expected.trace.supersteps[s].messages;
                EXPECT_EQ(messages.size(), originals.size());
                for (const auto& m : messages) {
                    auto match = std::find_if(originals.begin(), originals.end(), [&](const auto& o) {
                        return m.source == o.source && m.destination == o.destination && m.tag == o.tag;
                    });
                    ASSERT_NE(match, originals.end());
                    EXPECT_EQ(m.words, match->words);
                    EXPECT_EQ(m.region->row, match->region->row); EXPECT_EQ(m.region->col, match->region->col);
                }
            }
        }
        auto bad = std::span(output).first(1);
        EXPECT_THROW(plan.run_into(bad, *executor), std::invalid_argument);
    }
}

TEST(PreparedTest, RejectsForeignViewsDuplicateOrUndrainedMessagesAndMutableSealedPlan) {
    dmm::PreparedBsp plan(2);
    const auto a = plan.allocate(0, 4), b = plan.allocate(1, 4);
    const auto s = plan.step("replication");
    EXPECT_THROW(plan.copy(s, 0, {a,0,2}, {b,0,2}, 2,2), std::invalid_argument);
    EXPECT_THROW(plan.copy(s, 0, {a,0,2}, {a,0,2}, 2,2), std::invalid_argument);
    const auto msg = plan.send(s, 0,1,0,{a,0,2},{0,0,2,2});
    EXPECT_THROW(plan.send(s, 0,1,0,{a,0,2},{0,0,2,2}), std::invalid_argument);
    EXPECT_THROW(plan.seal(2,2,{1,1,1}), std::invalid_argument);
    const auto next = plan.step("reduction");
    EXPECT_THROW(plan.receive(next, 0,msg,{a,0,2}), std::invalid_argument);
    EXPECT_THROW(plan.receive(next, 1,msg,{b,3,2}), std::invalid_argument);
    plan.receive(next,1,msg,{b,0,2},true);
    EXPECT_THROW(plan.receive(next,1,msg,{b,0,2}), std::invalid_argument);
    std::fill(plan.buffer(a).begin(), plan.buffer(a).end(), 2.f);
    std::fill(plan.buffer(b).begin(), plan.buffer(b).end(), 3.f);
    plan.seal(2,2,{1,1,1});
    EXPECT_THROW(plan.allocate(0,1), std::logic_error);
    dmm::SerialExecutor executor;
    plan.execute(executor);
    EXPECT_EQ(plan.buffer(b)[0], 5.f);
    EXPECT_EQ(dmm::count(plan.trace()).total_flops, 4u);
}

TEST(PreparedTest, IndependentMicrobenchmarksMeasureRealWork) {
    dmm::ThreadExecutor executor(2);
    for (const auto& kind : {"compute", "copy", "synchronization"})
        EXPECT_GT(dmm::microbenchmark(kind, 32, 4, 2, executor), 0);
    EXPECT_THROW(dmm::microbenchmark("fake",32,4,2,executor), std::invalid_argument);
}
