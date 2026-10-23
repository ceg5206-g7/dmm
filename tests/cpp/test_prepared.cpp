#include "dmm/matmul.hpp"
#include "dmm/calibration.hpp"
#include "dmm/cannon.hpp"
#include "dmm/coverage.hpp"
#include "dmm/collectives.hpp"
#include "dmm/sharding.hpp"
#include <gtest/gtest.h>
#include <atomic>
#include <cstdlib>
#include <new>

namespace {
std::atomic<bool> watch_allocations{false};
std::atomic<std::size_t> allocations{0};
} // namespace
// Catch payload and metadata allocations on worker threads as well as the caller.
void* operator new(std::size_t bytes) {
    if (watch_allocations.load(std::memory_order_relaxed))
        ++allocations;
    if (auto* pointer = std::malloc(bytes ? bytes : 1))
        return pointer;
    throw std::bad_alloc();
}
void operator delete(void* pointer) noexcept { std::free(pointer); }
void* operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete[](void* pointer) noexcept { ::operator delete(pointer); }

TEST(PreparedTest, RepeatedPaddedProductsMatchReferenceTraceAndAllocateNothing) {
    for (auto P : {1u, 4u, 16u})
        for (auto workers : {0u, 2u}) {
            auto executor = dmm::make_executor(workers);
            dmm::Matrix<float> a(17), b(17);
            for (std::size_t i = 0; i < a.values.size(); ++i) {
                a.values[i] = static_cast<float>(static_cast<int>(i % 13) - 6) / 32;
                b.values[i] = static_cast<float>(static_cast<int>(i % 7) - 3) / 16;
            }
            const auto expected = dmm::Cannon{P, 1}.run(a, b, *executor);
            const auto ordinary = dmm::Cannon{P}.run(a, b, *executor);
            EXPECT_EQ(expected.product.values, ordinary.product.values);
            EXPECT_EQ(dmm::count(expected.trace), dmm::count(ordinary.trace));
            EXPECT_EQ((dmm::Cannon{P, 1}.matmul(a, b, *executor).values), expected.product.values);
            dmm::PreparedMatmul plan(a, b, P, 1, dmm::build_cannon);
            std::vector<float> output(a.values.size());
            EXPECT_THROW(plan.trace(), std::logic_error);
            for (int repeat = 0; repeat < 3; ++repeat) {
                allocations = 0;
                watch_allocations = true;
                const auto times = plan.run_into(output, *executor);
                watch_allocations = false;
                EXPECT_EQ(allocations.load(), 0u) << "P=" << P << ", H=" << workers;
                EXPECT_EQ(output, expected.product.values);
                EXPECT_GT(times.end_to_end.total, 0);
                EXPECT_NEAR(times.end_to_end.total,
                            times.end_to_end.compute + times.end_to_end.copy +
                                times.end_to_end.synchronization,
                            1e-10);
                const auto trace = plan.trace();
                EXPECT_NO_THROW(dmm::validate_coverage(trace));
                const auto actual = dmm::count(trace), original = dmm::count(expected.trace);
                EXPECT_EQ(actual, original);
                EXPECT_EQ(trace.peak_words, expected.trace.peak_words);
                for (std::size_t s = 0; s < trace.supersteps.size(); ++s) {
                    const auto& messages = trace.supersteps[s].messages;
                    const auto& originals = expected.trace.supersteps[s].messages;
                    EXPECT_EQ(trace.supersteps[s].peak_words, expected.trace.supersteps[s].peak_words);
                    EXPECT_EQ(messages.size(), originals.size());
                    for (const auto& m : messages) {
                        auto match = std::find_if(originals.begin(), originals.end(), [&](const auto& o) {
                            return m.source == o.source && m.destination == o.destination && m.tag == o.tag;
                        });
                        ASSERT_NE(match, originals.end());
                        EXPECT_EQ(m.words, match->words);
                        EXPECT_EQ(m.region->row, match->region->row);
                        EXPECT_EQ(m.region->col, match->region->col);
                    }
                }
            }
            auto bad = std::span(output).first(1);
            EXPECT_THROW(plan.run_into(bad, *executor), std::invalid_argument);
        }
}

TEST(PreparedTest, RejectsForeignViewsDuplicateOrUndrainedMessagesAndMutableSealedPlan) {
    dmm::BspPlan plan(2);
    const auto a = plan.allocate(0, 4), b = plan.allocate(1, 4);
    const auto s = plan.step("replication");
    EXPECT_THROW(plan.copy(s, 0, {a, 0, 2}, {b, 0, 2}, 2, 2), std::invalid_argument);
    EXPECT_THROW(plan.copy(s, 0, {a, 0, 2}, {a, 0, 2}, 2, 2), std::invalid_argument);
    const auto msg = plan.send(s, 0, 1, 0, {a, 0, 2}, {0, 0, 2, 2});
    EXPECT_THROW(plan.send(s, 0, 1, 0, {a, 0, 2}, {0, 0, 2, 2}), std::invalid_argument);
    EXPECT_THROW(plan.seal(2, 2, {1, 1, 1}), std::invalid_argument);
    const auto next = plan.step("reduction");
    EXPECT_THROW(plan.receive(next, 0, msg, {a, 0, 2}), std::invalid_argument);
    EXPECT_THROW(plan.receive(next, 1, msg, {b, 3, 2}), std::invalid_argument);
    plan.receive(next, 1, msg, {b, 0, 2}, true);
    EXPECT_THROW(plan.receive(next, 1, msg, {b, 0, 2}), std::invalid_argument);
    std::fill(plan.buffer(a).begin(), plan.buffer(a).end(), 2.f);
    std::fill(plan.buffer(b).begin(), plan.buffer(b).end(), 3.f);
    plan.seal(2, 2, {1, 1, 1});
    EXPECT_THROW(plan.allocate(0, 1), std::logic_error);
    dmm::SerialExecutor executor;
    plan.execute(executor);
    EXPECT_EQ(plan.buffer(b)[0], 5.f);
    EXPECT_EQ(dmm::count(plan.trace()).total_flops, 4u);
}

TEST(PreparedTest, IndependentMicrobenchmarksMeasureRealWork) {
    dmm::ThreadExecutor executor(2);
    for (const auto& kind : {"compute", "copy", "synchronization"})
        EXPECT_GT(dmm::microbenchmark(kind, 32, 4, 2, executor), 0);
    EXPECT_THROW(dmm::microbenchmark("fake", 32, 4, 2, executor), std::invalid_argument);
}

TEST(PreparedTest, FailedExecutorBatchCanRecoverWithCorrectMemoryAndOutput) {
    class FailingExecutor : public dmm::Executor {
        int calls = 0;
        void for_each(std::size_t n, const std::function<void(std::size_t)>& task) override {
            if (++calls == 4)
                throw std::runtime_error("interrupted after delivery");
            for (std::size_t i = 0; i < n; ++i)
                task(i);
        }
    } failing;
    dmm::Matrix<float> a(17);
    std::fill(a.values.begin(), a.values.end(), .25f);
    dmm::PreparedMatmul recovered(a, a, 4, 1, dmm::build_cannon), fresh(a, a, 4, 1, dmm::build_cannon);
    dmm::SerialExecutor executor;
    std::vector<float> output(a.values.size()), expected(a.values.size());
    EXPECT_THROW(recovered.run_into(output, failing), std::runtime_error);
    EXPECT_THROW(recovered.trace(), std::logic_error);
    recovered.run_into(output, executor);
    fresh.run_into(expected, executor);
    EXPECT_EQ(output, expected);
    EXPECT_EQ(recovered.trace().peak_words, fresh.trace().peak_words);
    EXPECT_EQ(dmm::count(recovered.trace()), dmm::count(fresh.trace()));
}

TEST(PreparedTest, SharedTreesMatchHandCountsOnDisjointUnevenGroupsWithoutAllocation) {
    const std::vector<dmm::TreeGroup> groups{{3, 1, 6, 0}, {2, 5}, {4}};
    const std::vector<dmm::ElementRegion> regions(3, {0, 0, 2, 3});
    dmm::BspPlan prepared(7);
    std::vector<std::size_t> buffers;
    for (std::size_t node = 0; node < 7; ++node) {
        buffers.push_back(prepared.allocate(node, 6));
        std::fill(prepared.buffer(buffers.back()).begin(), prepared.buffer(buffers.back()).end(),
                  static_cast<float>(node + 1));
    }
    dmm::tree_broadcast(prepared, groups, buffers, 9, regions);
    dmm::tree_reduce_sum(prepared, groups, buffers, 10, regions);
    prepared.seal(7, 7, {1, 1, 1});
    dmm::ThreadExecutor executor(2);
    allocations = 0;
    watch_allocations = true;
    prepared.execute(executor);
    watch_allocations = false;
    EXPECT_EQ(allocations.load(), 0u);
    const auto counts = dmm::count(prepared.trace());
    EXPECT_EQ(counts.supersteps, 6u);
    EXPECT_EQ(counts.messages, 8u);
    EXPECT_EQ(counts.total_words, 48u);
    EXPECT_EQ(counts.total_flops, 24u);
    for (auto node : groups[0])
        if (node != 0)
            EXPECT_GE(prepared.buffer(buffers[node])[0], 4.f);
    EXPECT_EQ(prepared.buffer(buffers[3])[0], 16.f);
    EXPECT_EQ(prepared.buffer(buffers[2])[0], 6.f);
}

TEST(PreparedTest, SharedPreparedTreeRejectsInvalidOwnersAndShapesBeforeAddingSteps) {
    dmm::BspPlan plan(2);
    const std::vector<std::size_t> buffers{plan.allocate(0, 4), plan.allocate(1, 4)};
    const std::vector<dmm::TreeGroup> groups{{0, 1}};
    const std::vector<dmm::ElementRegion> wrong{{0, 0, 1, 3}}, right{{0, 0, 2, 2}};
    EXPECT_THROW(dmm::tree_broadcast(plan, groups, buffers, 0, wrong), std::invalid_argument);
    auto foreign = buffers;
    std::swap(foreign[0], foreign[1]);
    EXPECT_THROW(dmm::tree_reduce_sum(plan, groups, foreign, 0, right), std::invalid_argument);
    EXPECT_THROW(plan.seal(2, 2, {1, 1, 1}), std::invalid_argument); // No step was added.
    dmm::tree_broadcast(plan, groups, buffers, 0, right);
    plan.seal(2, 2, {1, 1, 1});
}

TEST(PreparedTest, SharedPreparedRowConversionsCoverReplicatedBaseLayouts) {
    for (auto [P, c] : {std::pair{16u, 1u}, {64u, 4u}, {256u, 4u}}) {
        dmm::BspPlan plan(P);
        const auto n = dmm::padded_size(17, P), q = dmm::grid_size(P, c), height = n / P, block = n / q;
        std::vector<std::size_t> rows, tiles, output;
        dmm::Matrix<float> matrix(17);
        for (std::size_t i = 0; i < matrix.values.size(); ++i)
            matrix.values[i] = static_cast<float>(i);
        auto expected = [&](std::size_t row, std::size_t col) {
            return row < 17 && col < 17 ? matrix.values[row * 17 + col] : 0.f;
        };
        for (std::size_t node = 0; node < P; ++node) {
            rows.push_back(plan.allocate(node, n * height));
            tiles.push_back(plan.allocate(node, node < q * q ? block * block : 0));
            output.push_back(plan.allocate(node, n * height));
            for (std::size_t row = 0; row < height; ++row)
                for (std::size_t col = 0; col < n; ++col)
                    plan.buffer(rows[node])[row * n + col] = expected(node * height + row, col);
            std::fill(plan.buffer(output[node]).begin(), plan.buffer(output[node]).end(), -1.f);
        }
        dmm::rows_to_blocks(plan, n, c, rows, tiles, 0);
        dmm::blocks_to_rows(plan, n, c, tiles, output, 1);
        plan.seal(17, n, {q, q, q});
        dmm::SerialExecutor executor;
        allocations = 0;
        watch_allocations = true;
        plan.execute(executor);
        watch_allocations = false;
        EXPECT_EQ(allocations.load(), 0u);
        for (std::size_t node = 0; node < P; ++node) {
            for (std::size_t row = 0; row < height; ++row)
                for (std::size_t col = 0; col < n; ++col)
                    ASSERT_EQ(plan.buffer(output[node])[row * n + col], expected(node * height + row, col));
            if (node < q * q)
                for (std::size_t row = 0; row < block; ++row)
                    for (std::size_t col = 0; col < block; ++col)
                        ASSERT_EQ(plan.buffer(tiles[node])[row * block + col],
                                  expected(node / q * block + row, node % q * block + col));
        }
        const auto actual = dmm::count(plan.trace());
        const auto retained = c == 1 ? P : q; // Self segments: exactly the diagonal-aligned row nodes.
        EXPECT_EQ(actual.total_words, 2 * (P * q - retained) * height * block);
        EXPECT_EQ(actual.messages, 2 * (P * q - retained));
        EXPECT_EQ(actual.supersteps, 4u);
        EXPECT_EQ(actual.total_flops, 0u);
    }
}

TEST(PreparedTest, CommonLifecycleRunsAnIndependentBuilderAndRejectsMalformedRowBindings) {
    const auto local = +[](dmm::BspPlan& plan, dmm::MatmulLayout& layout) {
        const auto n = layout.n;
        layout.a = {plan.allocate(0, n * n)};
        layout.b = {plan.allocate(0, n * n)};
        layout.result = {plan.allocate(0, n * n)};
        layout.coverage = {1, 1, 1};
        const auto step = plan.step("multiply");
        plan.zero(step, 0, layout.result[0]);
        plan.multiply_add(step, 0, {0, 0, 0, n, n, n}, {layout.a[0], 0, n}, {layout.b[0], 0, n},
                          {layout.result[0], 0, n});
    };
    dmm::Matrix<float> a(3), b(3);
    for (std::size_t i = 0; i < 9; ++i)
        a.values[i] = float(i + 1);
    for (std::size_t i = 0; i < 3; ++i)
        b.values[i * 3 + i] = 1;
    dmm::PreparedMatmul job(a, b, 1, 1, local);
    a.values[0] = -99;
    dmm::SerialExecutor executor;
    std::vector<float> output(9);
    for (int repeat = 0; repeat < 2; ++repeat) {
        allocations = 0;
        watch_allocations = true;
        job.run_into(output, executor);
        watch_allocations = false;
        EXPECT_EQ(allocations.load(), 0u);
        for (std::size_t i = 0; i < 9; ++i)
            EXPECT_EQ(output[i], float(i + 1));
    }
    EXPECT_NO_THROW(dmm::validate_coverage(job.trace()));
    EXPECT_EQ(dmm::count(job.trace()).supersteps, 1u);
    EXPECT_THROW(dmm::PreparedMatmul(a, b, 1, 1, nullptr), std::invalid_argument);
    const auto missing = +[](dmm::BspPlan&, dmm::MatmulLayout&) {};
    EXPECT_THROW(dmm::PreparedMatmul(a, b, 1, 1, missing), std::invalid_argument);
    const auto aliased = +[](dmm::BspPlan& plan, dmm::MatmulLayout& layout) {
        auto id = plan.allocate(0, layout.n * layout.n);
        layout.a = {id};
        layout.b = {id};
        layout.result = {id};
    };
    EXPECT_THROW(dmm::PreparedMatmul(a, b, 1, 1, aliased), std::invalid_argument);
    const auto short_rows = +[](dmm::BspPlan& plan, dmm::MatmulLayout& layout) {
        layout.a = {plan.allocate(0, 1)};
        layout.b = {plan.allocate(0, 1)};
        layout.result = {plan.allocate(0, 1)};
    };
    EXPECT_THROW(dmm::PreparedMatmul(a, b, 1, 1, short_rows), std::invalid_argument);
}
