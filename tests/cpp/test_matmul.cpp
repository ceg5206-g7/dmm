#include "dmm/algorithms.hpp"
#include "dmm/calibration.hpp"
#include "dmm/validation.hpp"
#include "allocation_watch.hpp"
#include <gtest/gtest.h>

TEST(MatmulTest, OrdinaryAndPreparedUseIdenticalPlansWithNoExecutionAllocations) {
    for (auto P : {1u, 4u, 16u})
        for (auto workers : {0u, 2u}) {
            auto executor = dmm::make_executor(workers);
            dmm::Matrix<float> a(17), b(17);
            for (std::size_t i = 0; i < a.values.size(); ++i) {
                a.values[i] = float(int(i % 13) - 6) / 32;
                b.values[i] = float(int(i % 7) - 3) / 16;
            }
            const auto expected = dmm::matmul_traced(a, b, *executor, "cannon", P);
            EXPECT_EQ(dmm::matmul(a, b, *executor, "cannon", P).values, expected.product.values);
            dmm::PreparedMatmul prepared(a, b, P, 1, dmm::build_cannon);
            std::vector<float> output(a.values.size());
            EXPECT_THROW(prepared.trace(), std::logic_error);
            for (int repeat = 0; repeat < 3; ++repeat) {
                allocation_test::Watch watch;
                const auto times = prepared.run_into(output, *executor);
                const auto allocations = watch.finish();
                EXPECT_EQ(allocations, 0u) << "P=" << P << ", H=" << workers;
                EXPECT_EQ(output, expected.product.values);
                EXPECT_GT(times.end_to_end.total, 0);
                EXPECT_NEAR(times.end_to_end.total,
                            times.end_to_end.compute + times.end_to_end.copy +
                                times.end_to_end.synchronization,
                            1e-10);
                const auto trace = prepared.trace();
                EXPECT_NO_THROW(dmm::validate_coverage(trace));
                EXPECT_EQ(dmm::count(trace), dmm::count(expected.trace));
                EXPECT_EQ(trace.peak_words, expected.trace.peak_words);
                ASSERT_EQ(trace.supersteps.size(), expected.trace.supersteps.size());
                for (std::size_t s = 0; s < trace.supersteps.size(); ++s) {
                    const auto& actual = trace.supersteps[s];
                    const auto& original = expected.trace.supersteps[s];
                    EXPECT_EQ(actual.flops, original.flops);
                    EXPECT_EQ(actual.peak_words, original.peak_words);
                    ASSERT_EQ(actual.messages.size(), original.messages.size());
                    for (std::size_t j = 0; j < actual.messages.size(); ++j) {
                        const auto& m = actual.messages[j];
                        const auto& o = original.messages[j];
                        EXPECT_EQ(m.source, o.source);
                        EXPECT_EQ(m.destination, o.destination);
                        EXPECT_EQ(m.words, o.words);
                        ASSERT_EQ(m.regions.size(), o.regions.size());
                        for (std::size_t k = 0; k < m.regions.size(); ++k) {
                            EXPECT_EQ(m.regions[k].row, o.regions[k].row);
                            EXPECT_EQ(m.regions[k].col, o.regions[k].col);
                            EXPECT_EQ(m.regions[k].rows, o.regions[k].rows);
                            EXPECT_EQ(m.regions[k].cols, o.regions[k].cols);
                        }
                    }
                }
            }
            EXPECT_THROW(prepared.run_into(std::span(output).first(1), *executor), std::invalid_argument);
        }
}
TEST(MatmulTest, FailedExecutorBatchCanRecoverWithCorrectMemoryAndOutput) {
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
TEST(MatmulTest, IndependentBuilderUsesCommonLifecycleAndOwnedInputsWithStridedBindings) {
    const auto local = +[](dmm::BspPlan& plan, const dmm::MatmulConfig& config) {
        const auto n = config.padded_size;
        auto view = [&] {
            return plan.matrix_view(plan.allocate_buffer(0, n * (n + 2) + 1), n, n, 1, n + 2);
        };
        auto a = view(), b = view(), c = view();
        auto step = plan.add_superstep(dmm::Phase::Multiply);
        step.add_zero(c);
        step.add_gemm(a, b, c, {0, 0, 0});
        return dmm::MatmulDefinition{{a}, {b}, {c}, {1, 1, 1}};
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
        allocation_test::Watch watch;
        job.run_into(output, executor);
        auto allocations = watch.finish();
        EXPECT_EQ(allocations, 0u);
        for (std::size_t i = 0; i < 9; ++i)
            EXPECT_EQ(output[i], float(i + 1));
    }
    EXPECT_NO_THROW(dmm::validate_coverage(job.trace()));
    EXPECT_EQ(dmm::count(job.trace()).supersteps, 1u);
    EXPECT_THROW(dmm::PreparedMatmul(a, b, 1, 1, nullptr), std::invalid_argument);
    const auto missing = +[](dmm::BspPlan&, const dmm::MatmulConfig&) { return dmm::MatmulDefinition{}; };
    EXPECT_THROW(dmm::PreparedMatmul(a, b, 1, 1, missing), std::invalid_argument);
    const auto aliased = +[](dmm::BspPlan& plan, const dmm::MatmulConfig& config) {
        auto view = plan.matrix_view(plan.allocate_buffer(0, config.padded_size * config.padded_size),
                                     config.padded_size, config.padded_size);
        return dmm::MatmulDefinition{{view}, {view}, {view}, {1, 1, 1}};
    };
    EXPECT_THROW(dmm::PreparedMatmul(a, b, 1, 1, aliased), std::invalid_argument);
    const auto short_rows = +[](dmm::BspPlan& plan, const dmm::MatmulConfig&) {
        auto view = [&] { return plan.matrix_view(plan.allocate_buffer(0, 1), 1, 1); };
        return dmm::MatmulDefinition{{view()}, {view()}, {view()}, {1, 1, 1}};
    };
    EXPECT_THROW(dmm::PreparedMatmul(a, b, 1, 1, short_rows), std::invalid_argument);
}
TEST(MatmulTest, RegistrySelectsEveryLayoutAndRejectsUnknownNames) {
    EXPECT_EQ(dmm::select_builder("cannon"), dmm::build_cannon);
    EXPECT_EQ(dmm::select_builder("ring"), dmm::build_ring);
    EXPECT_EQ(dmm::select_builder("2.5d"), dmm::build_two_point_five_d);
    EXPECT_EQ(dmm::select_builder("2.5d-direct"), dmm::build_two_point_five_d_direct);
    for (const auto* layout : {"fox", "ring2", ""})
        EXPECT_THROW(dmm::select_builder(layout), std::invalid_argument);
}
TEST(CalibrationTest, IndependentMicrobenchmarksMeasureRealWork) {
    dmm::ThreadExecutor executor(2);
    for (const auto* kind : {"compute", "copy", "synchronization"})
        EXPECT_GT(dmm::microbenchmark(kind, 32, 4, 2, executor), 0);
    EXPECT_THROW(dmm::microbenchmark("fake", 32, 4, 2, executor), std::invalid_argument);
}
