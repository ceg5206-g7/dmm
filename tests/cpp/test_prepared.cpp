#include "dmm/prepared.hpp"
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


TEST(PreparedTest, FailedExecutorBatchCanRecoverWithCorrectMemoryAndOutput) {
    class FailingExecutor : public dmm::Executor {
        int calls = 0;
        void for_each(std::size_t n, const std::function<void(std::size_t)>& task) override {
            if (++calls == 4) throw std::runtime_error("interrupted after delivery");
            for (std::size_t i = 0; i < n; ++i) task(i);
        }
    } failing;
    dmm::Matrix<float> a(17);
    std::fill(a.values.begin(), a.values.end(), .25f);
    dmm::PreparedCannon recovered(a,a,4), fresh(a,a,4);
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

TEST(PreparedTest, SharedPreparedTreesMatchRuntimeOnDisjointUnevenGroupsWithoutAllocation) {
    const std::vector<dmm::TreeGroup> groups{{3,1,6,0}, {2,5}, {4}};
    const std::vector<dmm::ElementRegion> regions(3, {0,0,2,3});
    dmm::PreparedBsp prepared(7);
    dmm::Bsp runtime(7);
    auto values = runtime.buffers(6);
    std::vector<std::size_t> buffers;
    for (std::size_t node = 0; node < 7; ++node) {
        buffers.push_back(prepared.allocate(node, 6));
        std::fill(prepared.buffer(buffers.back()).begin(), prepared.buffer(buffers.back()).end(), static_cast<float>(node+1));
        values[node].assign(6, static_cast<float>(node+1));
    }
    dmm::tree_broadcast(prepared, groups, buffers, 9, regions);
    dmm::tree_reduce_sum(prepared, groups, buffers, 10, regions);
    prepared.seal(7,7,{1,1,1});
    dmm::ThreadExecutor executor(2);
    allocations = 0; watch_allocations = true;
    prepared.execute(executor);
    watch_allocations = false;
    EXPECT_EQ(allocations.load(), 0u);
    dmm::tree_broadcast(runtime, executor, groups, values, 9, regions);
    dmm::tree_reduce_sum(runtime, executor, groups, values, 10, regions);
    auto expected = dmm::count(runtime.finish()), actual = dmm::count(prepared.trace());
    actual.peak_words = expected.peak_words;
    EXPECT_EQ(actual, expected);
    for (std::size_t node = 0; node < 7; ++node)
        EXPECT_TRUE(std::equal(prepared.buffer(buffers[node]).begin(), prepared.buffer(buffers[node]).end(), values[node].begin()));
    EXPECT_EQ(prepared.buffer(buffers[3])[0], 16.f);
    EXPECT_EQ(prepared.buffer(buffers[2])[0], 6.f);
}

TEST(PreparedTest, SharedPreparedTreeRejectsInvalidOwnersAndShapesBeforeAddingSteps) {
    dmm::PreparedBsp plan(2);
    const std::vector<std::size_t> buffers{plan.allocate(0,4), plan.allocate(1,4)};
    const std::vector<dmm::TreeGroup> groups{{0,1}};
    const std::vector<dmm::ElementRegion> wrong{{0,0,1,3}}, right{{0,0,2,2}};
    EXPECT_THROW(dmm::tree_broadcast(plan, groups, buffers, 0, wrong), std::invalid_argument);
    auto foreign = buffers; std::swap(foreign[0],foreign[1]);
    EXPECT_THROW(dmm::tree_reduce_sum(plan, groups, foreign, 0, right), std::invalid_argument);
    EXPECT_THROW(plan.seal(2,2,{1,1,1}), std::invalid_argument); // No step was added.
    dmm::tree_broadcast(plan, groups, buffers, 0, right);
    plan.seal(2,2,{1,1,1});
}

TEST(PreparedTest, SharedPreparedRowConversionsCoverReplicatedBaseLayouts) {
    for (auto [P,c] : {std::pair{16u,1u}, {64u,4u}, {256u,4u}}) {
        dmm::PreparedBsp plan(P);
        const auto n = dmm::padded_size(17,P), q = dmm::grid_size(P,c), height = n/P, block = n/q;
        std::vector<std::size_t> rows, tiles, output;
        dmm::Matrix<float> matrix(17);
        for (std::size_t i = 0; i < matrix.values.size(); ++i) matrix.values[i] = static_cast<float>(i);
        auto expected = [&](std::size_t row, std::size_t col) { return row < 17 && col < 17 ? matrix.values[row*17+col] : 0.f; };
        for (std::size_t node = 0; node < P; ++node) {
            rows.push_back(plan.allocate(node,n*height));
            tiles.push_back(plan.allocate(node,node < q*q ? block*block : 0));
            output.push_back(plan.allocate(node,n*height));
            for (std::size_t row = 0; row < height; ++row) for (std::size_t col = 0; col < n; ++col)
                plan.buffer(rows[node])[row*n+col] = expected(node*height+row,col);
            std::fill(plan.buffer(output[node]).begin(), plan.buffer(output[node]).end(), -1.f);
        }
        dmm::rows_to_blocks(plan,n,c,rows,tiles,0);
        dmm::blocks_to_rows(plan,n,c,tiles,output,1);
        plan.seal(17,n,{q,q,q});
        dmm::SerialExecutor executor;
        allocations = 0; watch_allocations = true;
        plan.execute(executor);
        watch_allocations = false;
        EXPECT_EQ(allocations.load(),0u);
        for (std::size_t node = 0; node < P; ++node) {
            for (std::size_t row = 0; row < height; ++row) for (std::size_t col = 0; col < n; ++col)
                ASSERT_EQ(plan.buffer(output[node])[row*n+col],expected(node*height+row,col));
            if (node < q*q) for (std::size_t row = 0; row < block; ++row) for (std::size_t col = 0; col < block; ++col)
                ASSERT_EQ(plan.buffer(tiles[node])[row*block+col],expected(node/q*block+row,node%q*block+col));
        }
        dmm::Bsp runtime(P);
        const auto source = dmm::make_row_shards(runtime,matrix);
        const auto native = dmm::rows_to_blocks(runtime,executor,source,c,0);
        const auto gathered = dmm::blocks_to_rows(runtime,executor,native,1);
        auto original = dmm::count(runtime.finish()), actual = dmm::count(plan.trace());
        actual.peak_words = original.peak_words;
        EXPECT_EQ(actual,original);
    }
}
