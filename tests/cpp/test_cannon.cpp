#include "dmm/algorithms.hpp"
#include "dmm/validation.hpp"
#include "dmm/redistribution.hpp"
#include <gtest/gtest.h>
#include "test_counts.hpp"
#include <algorithm>
#include <cstddef>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
dmm::Matrix<double> reference(const dmm::Matrix<float>& a, const dmm::Matrix<float>& b) {
    const auto n = a.size;
    dmm::Matrix<double> c(n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            for (std::size_t k = 0; k < n; ++k)
                c.values[i * n + j] += static_cast<double>(a.values[i * n + k]) * b.values[k * n + j];
    return c;
}

dmm::Matrix<double> constant_reference(std::size_t size, double value) {
    dmm::Matrix<double> matrix(size);
    std::fill(matrix.values.begin(), matrix.values.end(), value);
    return matrix;
}

class CannonTest : public ::testing::TestWithParam<std::size_t> {};

TEST_P(CannonTest, MatchesFP64ReferenceAndPreservesInputsAcrossSizes) {
    auto executor = dmm::make_executor(GetParam());
    std::mt19937 random(5206);
    std::uniform_real_distribution<float> value(-1, 1);
    for (std::size_t n : {4, 8, 12, 16, 132}) {
        SCOPED_TRACE(::testing::Message() << "size=" << n);
        dmm::Matrix<float> a(n), b(n);
        for (auto& x : a.values)
            x = value(random);
        for (auto& x : b.values)
            x = value(random);
        const auto a_before = a.values, b_before = b.values;
        const auto expected = reference(a, b);
        const auto result = dmm::matmul(a, b, *executor);
        EXPECT_NO_THROW(dmm::validate(result, expected));
        const auto repeated = dmm::matmul(a, b, *executor);
        EXPECT_EQ(result.values, repeated.values);
        EXPECT_EQ(a_before, a.values);
        EXPECT_EQ(b_before, b.values);
    }
}

TEST_P(CannonTest, HandlesZeroIdentityOnesAndCancellation) {
    auto executor = dmm::make_executor(GetParam());
    dmm::Matrix<float> a(4), identity(4);
    for (std::size_t i = 0; i < 4; ++i)
        identity.values[i * 4 + i] = 1;
    EXPECT_NO_THROW(dmm::validate(dmm::matmul(a, identity, *executor), constant_reference(4, 0)));
    std::fill(a.values.begin(), a.values.end(), 1);
    EXPECT_NO_THROW(dmm::validate(dmm::matmul(a, identity, *executor), constant_reference(4, 1)));
    std::fill(identity.values.begin(), identity.values.end(), 1);
    EXPECT_NO_THROW(dmm::validate(dmm::matmul(a, identity, *executor), constant_reference(4, 4)));
    for (std::size_t k = 0; k < 4; ++k)
        for (std::size_t j = 0; j < 4; ++j)
            identity.values[k * 4 + j] = k % 2 ? -1 : 1;
    EXPECT_NO_THROW(dmm::validate(dmm::matmul(a, identity, *executor), constant_reference(4, 0)));
}

TEST_P(CannonTest, PreservesTileAccumulationOrder) {
    auto executor = dmm::make_executor(GetParam());
    dmm::Matrix<float> a(16), b(16);
    for (std::size_t i = 0; i < 16; ++i) {
        a.values[i * 16] = 1e8f;
        a.values[i * 16 + 4] = 1;
        a.values[i * 16 + 8] = -1e8f;
        a.values[i * 16 + 12] = 1;
    }
    std::fill(b.values.begin(), b.values.end(), 1);
    const auto result = dmm::matmul_traced(a, b, *executor).product;
    for (std::size_t i = 0; i < 16; ++i)
        for (std::size_t j = 0; j < 16; ++j)
            EXPECT_EQ(result.values[i * 16 + j], (i / 4 + j / 4) % 2 == 0 ? 1.f : 0.f);
}

INSTANTIATE_TEST_SUITE_P(Workers, CannonTest, ::testing::Values(0, 1, 2, 5),
                         ::testing::PrintToStringParamName());

TEST(CannonValidationTest, RejectsMismatchedUnsupportedAndMalformedMatrices) {
    dmm::SerialExecutor executor;
    dmm::Matrix<float> a(4), b(3);
    EXPECT_THROW(dmm::matmul(a, b, executor), std::invalid_argument);
    a.values.pop_back();
    EXPECT_THROW(dmm::matmul(a, a, executor), std::invalid_argument);
}

TEST(CannonDispatchTest, UsesThreePreparedBatchesPerLogicalSuperstep) {
    struct RecordingExecutor final : dmm::Executor {
        std::vector<std::size_t> batches;
        void for_each(std::size_t count, const std::function<void(std::size_t)>& task) override {
            batches.push_back(count);
            for (std::size_t i = 0; i < count; ++i)
                task(i);
        }
    } executor;
    dmm::Matrix<float> a(4), b(4);
    std::fill(a.values.begin(), a.values.end(), 1);
    std::fill(b.values.begin(), b.values.end(), 1);
    const auto result = dmm::matmul_traced(a, b, executor);
    EXPECT_NO_THROW(dmm::validate(result.product, constant_reference(4, 4)));
    // One A+B conversion exchange, the skew (which also consumes the conversion
    // messages), four rounds, then the output conversion's send and receive.
    EXPECT_EQ(dmm::count(result.trace).supersteps, 8u);
    EXPECT_EQ(executor.batches, (std::vector<std::size_t>(3 * 8, 16)));
}

class CannonTraceTest : public ::testing::TestWithParam<std::size_t> {};

// Hand count for block size b: the skew moves 12 A and 12 B blocks; each of
// three shifts moves all 32 blocks; every node does 2b^3 FLOPs per round.
TEST_P(CannonTraceTest, MatchesHandComputedCommunication) {
    auto executor = dmm::make_executor(GetParam());
    for (std::size_t n : {16, 32, 48}) {
        SCOPED_TRACE(::testing::Message() << "size=" << n);
        dmm::Matrix<float> a(n), b(n);
        std::fill(a.values.begin(), a.values.end(), 1);
        std::fill(b.values.begin(), b.values.end(), 1);
        auto [product, trace] = dmm::matmul_traced(a, b, *executor);
        EXPECT_NO_THROW(dmm::validate(product, constant_reference(n, static_cast<double>(n))));
        std::erase_if(trace.supersteps, [](const auto& step) {
            return step.label == "input_conversion" || step.label == "output_conversion";
        });
        const auto block = n / 4, words = block * block, flops = 2 * words * block;
        ASSERT_EQ(trace.nodes, 16u);
        ASSERT_EQ(trace.supersteps.size(), 5u);
        expect_work_and_communication(
            dmm::count(trace), (dmm::Counts{5, 16 * 4 * flops, 4 * flops, 120 * words, 8 * words, 120}));
        expect_work_and_communication(dmm::count(trace.supersteps[0]),
                                      (dmm::Counts{1, 0, 0, 24 * words, 2 * words, 24}));
        for (std::size_t s = 1; s < 4; ++s)
            expect_work_and_communication(dmm::count(trace.supersteps[s]),
                                          (dmm::Counts{1, 16 * flops, flops, 32 * words, 2 * words, 32}));
        expect_work_and_communication(dmm::count(trace.supersteps[4]),
                                      (dmm::Counts{1, 16 * flops, flops, 0, 0, 0}));
        EXPECT_EQ(trace.supersteps[0].label, "skew");
        for (std::size_t s = 1; s < 5; ++s)
            EXPECT_EQ(trace.supersteps[s].label, "multiply");
    }
}

TEST_P(CannonTraceTest, SkewsAndShiftsAlongGridRowsAndColumns) {
    auto executor = dmm::make_executor(GetParam());
    dmm::Matrix<float> a(16), b(16);
    auto trace = dmm::matmul_traced(a, b, *executor).trace;
    std::erase_if(trace.supersteps, [](const auto& step) {
        return step.label == "input_conversion" || step.label == "output_conversion";
    });
    const auto messages_from = [&](std::size_t step, std::size_t source) {
        std::vector<std::size_t> sent;
        for (const auto& message : trace.supersteps[step].messages)
            if (message.source == source) {
                EXPECT_EQ(message.words, 16u);
                sent.push_back(message.destination);
                EXPECT_EQ(message.regions.size(), 1u);
            }
        return sent;
    };
    using Sent = std::vector<std::size_t>;
    // Node (1, 2) = 6 skews A to (1, 1) = 5 and B to (3, 2) = 14, then shifts to (1, 1) and (0, 2).
    EXPECT_EQ(messages_from(0, 6), (Sent{5, 14}));
    EXPECT_EQ(messages_from(1, 6), (Sent{5, 2}));
    // Node (0, 0) keeps both blocks during the skew.
    EXPECT_TRUE(messages_from(0, 0).empty());
    EXPECT_EQ(messages_from(3, 0), (Sent{3, 12}));
    EXPECT_TRUE(messages_from(4, 0).empty());
    const auto& skew_messages = trace.supersteps[0].messages;
    const auto& first_round = trace.supersteps[1].messages;
    for (const auto& m : skew_messages)
        if (m.source == 6) {
            ASSERT_EQ(m.regions.size(), 1u);
            EXPECT_EQ(m.regions[0].row, 4u);
            EXPECT_EQ(m.regions[0].col, 8u);
        }
    for (const auto& m : first_round)
        if (m.source == 6) {
            ASSERT_EQ(m.regions.size(), 1u);
            EXPECT_EQ(m.regions[0].row, m.destination == 5 ? 4u : 12u);
            EXPECT_EQ(m.regions[0].col, m.destination == 5 ? 12u : 8u);
        }
}

INSTANTIATE_TEST_SUITE_P(Workers, CannonTraceTest, ::testing::Values(0, 2),
                         ::testing::PrintToStringParamName());

TEST(CannonStateTest, ResultsAreIndependentAcrossCallsAndExecutors) {
    dmm::ThreadExecutor executor(2);
    dmm::Matrix<float> a(4), b(4);
    std::fill(a.values.begin(), a.values.end(), 1);
    std::fill(b.values.begin(), b.values.end(), 1);
    const auto first = dmm::matmul(a, b, executor);
    std::fill(a.values.begin(), a.values.end(), 2);
    const auto second = dmm::matmul(a, b, executor);
    dmm::SerialExecutor serial;
    EXPECT_NO_THROW(dmm::validate(dmm::matmul(a, b, serial), constant_reference(4, 8)));
    EXPECT_NO_THROW(dmm::validate(first, constant_reference(4, 4)));
    EXPECT_NO_THROW(dmm::validate(second, constant_reference(4, 8)));
    bool executed = false;
    EXPECT_NO_THROW(executor.for_each(1, [&](std::size_t) { executed = true; }));
    EXPECT_TRUE(executed);
}
} // namespace

TEST(ConfigurableCannonTest, MatchesReferenceAndCountsAcrossGrids) {
    dmm::ThreadExecutor executor(2);
    dmm::Matrix<float> a(32), b(32);
    std::mt19937 random(5206);
    std::uniform_real_distribution<float> value(-1, 1);
    for (auto& x : a.values)
        x = value(random);
    for (auto& x : b.values)
        x = value(random);
    const auto expected = reference(a, b);
    for (std::size_t P : {1, 4, 16, 64, 256}) {
        SCOPED_TRACE(P);
        const auto n = dmm::padded_size(32, P), q = dmm::grid_size(P), block = n / q, words = block * block;
        const auto result = dmm::matmul_traced(a, b, executor, "cannon", P, 1);
        EXPECT_NO_THROW(dmm::validate(result.product, expected));
        EXPECT_EQ(result.trace.nodes, P);
        const auto counts = dmm::count_native(result.trace);
        // P = 1 has nothing to skew: zeroing C joins the single multiply step.
        EXPECT_EQ(counts.supersteps, q == 1 ? 1 : q + 1);
        EXPECT_EQ(counts.total_flops, 2 * n * n * n);
        EXPECT_EQ(counts.critical_flops, counts.total_flops / P);
        EXPECT_EQ(counts.messages, 2 * (q - 1) * (q + P));
        EXPECT_EQ(counts.total_words, counts.messages * words);
        EXPECT_EQ(counts.critical_words, q == 1 ? 0 : 2 * q * words);
    }
}

// P = 1 folds the zeroing skew into the single multiply superstep; the merged
// copy batch must clear C before the GEMM accumulates into it.
TEST(ConfigurableCannonTest, MergedZeroingRunsBeforeTheMultiplyStep) {
    constexpr std::size_t n = 4;
    dmm::BspPlan plan(1);
    const auto definition = dmm::build_cannon(plan, {n, 1, 1});
    plan.finalize(n, n, definition.product_grid, true);
    const auto fill_rows = [&](dmm::MatrixView view, float value) {
        auto data = plan.buffer(view.buffer).subspan(view.offset);
        for (std::size_t row = 0; row < view.rows; ++row)
            std::fill_n(data.data() + row * view.stride, view.cols, value);
    };
    fill_rows(definition.input_a_rows.front(), 1.f);
    fill_rows(definition.input_b_rows.front(), 1.f);
    fill_rows(definition.output_c_rows.front(), 1e9f);
    dmm::SerialExecutor executor;
    plan.execute(executor);
    EXPECT_EQ(dmm::count(plan.trace()).supersteps, 1u);
    std::vector<float> product(n * n);
    const auto view = definition.output_c_rows.front();
    const auto data = plan.buffer(view.buffer);
    for (std::size_t row = 0; row < view.rows; ++row)
        for (std::size_t col = 0; col < view.cols; ++col)
            product[row * n + col] = data[view.offset + row * view.stride + col];
    EXPECT_EQ(product, (std::vector<float>(n * n, static_cast<float>(n))));
}

TEST(GridConfigTest, ValidatesTwoAndTwoPointFiveDimensionalGrids) {
    EXPECT_EQ(dmm::grid_size(64, 4), 4u);
    EXPECT_EQ(dmm::grid_size(256, 4), 8u);
    dmm::SerialExecutor executor;
    dmm::Matrix<float> a(32);
    for (const auto [P, c] :
         std::vector<std::pair<std::size_t, std::size_t>>{{0, 1}, {16, 0}, {3, 1}, {16, 4}, {64, 2}, {65, 4}})
        EXPECT_THROW(dmm::grid_size(P, c), std::invalid_argument);
    EXPECT_THROW((dmm::matmul_traced(a, a, executor, "cannon", 64, 4)), std::invalid_argument);
}

TEST(CommonCannonTest, PaddingCroppingAndEndToEndCountsAreRealExecutedWork) {
    dmm::ThreadExecutor executor(2);
    std::mt19937 random(5206);
    std::uniform_real_distribution<float> value(-1, 1);
    for (std::size_t n : {1, 17, 65, 257}) {
        dmm::Matrix<float> a(n), b(n);
        for (auto& x : a.values)
            x = value(random);
        for (auto& x : b.values)
            x = value(random);
        const auto result = dmm::matmul_traced(a, b, executor);
        EXPECT_EQ(result.product.size, n);
        const auto padded = dmm::padded_size(n, 16);
        EXPECT_EQ(result.trace.original_size, n);
        EXPECT_EQ(result.trace.padded_size, padded);
        EXPECT_EQ(dmm::count(result.trace).total_flops, 2 * padded * padded * padded);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) {
                double expected = 0;
                for (std::size_t k = 0; k < n; ++k)
                    expected += static_cast<double>(a.values[i * n + k]) * b.values[k * n + j];
                EXPECT_NEAR(result.product.values[i * n + j], expected, 1e-4 + 1e-3 * std::abs(expected));
            }
    }
}
