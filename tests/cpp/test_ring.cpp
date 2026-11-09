#include "dmm/algorithms.hpp"
#include "dmm/validation.hpp"
#include <gtest/gtest.h>
#include "test_counts.hpp"
#include <algorithm>
#include <cstddef>
#include <limits>
#include <random>
#include <stdexcept>
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

class RingTest : public ::testing::TestWithParam<std::size_t> {};

TEST_P(RingTest, MatchesFP64ReferenceAcrossSizesAndGrids) {
    auto executor = dmm::make_executor(GetParam());
    std::mt19937 random(5206);
    std::uniform_real_distribution<float> value(-1, 1);
    for (std::size_t P : {1, 2, 3, 4, 16})
        for (std::size_t n : {1, 4, 5, 12, 33}) {
            SCOPED_TRACE(::testing::Message() << "P=" << P << " size=" << n);
            dmm::Matrix<float> a(n), b(n);
            for (auto& x : a.values)
                x = value(random);
            for (auto& x : b.values)
                x = value(random);
            const auto a_before = a.values, b_before = b.values;
            const auto expected = reference(a, b);
            const auto result = dmm::matmul(a, b, *executor, "ring", P, 1);
            EXPECT_NO_THROW(dmm::validate(result, expected));
            EXPECT_EQ(a_before, a.values);
            EXPECT_EQ(b_before, b.values);
        }
}

TEST_P(RingTest, HandlesZeroIdentityOnesAndCancellation) {
    auto executor = dmm::make_executor(GetParam());
    dmm::Matrix<float> a(4), identity(4);
    for (std::size_t i = 0; i < 4; ++i)
        identity.values[i * 4 + i] = 1;
    EXPECT_NO_THROW(dmm::validate(dmm::matmul(a, identity, *executor, "ring", 4, 1),
                                  constant_reference(4, 0)));
    std::fill(a.values.begin(), a.values.end(), 1);
    EXPECT_NO_THROW(dmm::validate(dmm::matmul(a, identity, *executor, "ring", 4, 1),
                                  constant_reference(4, 1)));
    std::fill(identity.values.begin(), identity.values.end(), 1);
    EXPECT_NO_THROW(dmm::validate(dmm::matmul(a, identity, *executor, "ring", 4, 1),
                                  constant_reference(4, 4)));
    for (std::size_t k = 0; k < 4; ++k)
        for (std::size_t j = 0; j < 4; ++j)
            identity.values[k * 4 + j] = k % 2 ? -1 : 1;
    EXPECT_NO_THROW(dmm::validate(dmm::matmul(a, identity, *executor, "ring", 4, 1),
                                  constant_reference(4, 0)));
}

// Hand count for row height h: P multiply supersteps; rounds 0..P-2 each send
// one h*N block per node to the right neighbour, so P(P-1) messages of h*N
// words; every node does 2*h*h*N FLOPs per round.
TEST_P(RingTest, MatchesHandComputedCommunication) {
    auto executor = dmm::make_executor(GetParam());
    for (std::size_t P : {1, 2, 4, 16}) {
        SCOPED_TRACE(::testing::Message() << "P=" << P);
        dmm::Matrix<float> a(32), b(32);
        std::fill(a.values.begin(), a.values.end(), 1);
        std::fill(b.values.begin(), b.values.end(), 1);
        auto [product, trace] = dmm::matmul_traced(a, b, *executor, "ring", P, 1);
        EXPECT_NO_THROW(dmm::validate(product, constant_reference(32, 32)));
        const auto height = 32 / P, words = height * 32, flops = 2 * height * height * 32;
        ASSERT_EQ(trace.nodes, P);
        ASSERT_EQ(trace.supersteps.size(), P);
        for (const auto& step : trace.supersteps)
            EXPECT_EQ(step.label, "multiply");
        expect_work_and_communication(dmm::count(trace),
                                      (dmm::Counts{P, P * P * flops, P * flops, P * (P - 1) * words,
                                                   (P - 1) * words, P * (P - 1)}));
        for (std::size_t s = 0; s + 1 < P; ++s)
            expect_work_and_communication(dmm::count(trace.supersteps[s]),
                                          (dmm::Counts{1, P * flops, flops, P * words, words, P}));
        expect_work_and_communication(dmm::count(trace.supersteps[P - 1]),
                                      (dmm::Counts{1, P * flops, flops, 0, 0, 0}));
    }
}

TEST_P(RingTest, CoversEveryInnerProductOnce) {
    auto executor = dmm::make_executor(GetParam());
    for (std::size_t P : {2, 3, 4}) {
        SCOPED_TRACE(::testing::Message() << "P=" << P);
        dmm::Matrix<float> a(12), b(12);
        std::fill(a.values.begin(), a.values.end(), 1);
        std::fill(b.values.begin(), b.values.end(), 1);
        const auto trace = dmm::matmul_traced(a, b, *executor, "ring", P, 1).trace;
        std::vector<std::size_t> seen(P * P);
        for (const auto& step : trace.supersteps)
            for (const auto& product : step.products) {
                ASSERT_EQ(product.product.j, 0u);
                ASSERT_LT(product.product.i, P);
                ASSERT_LT(product.product.k, P);
                EXPECT_EQ(product.product.m, 12 / P);
                EXPECT_EQ(product.product.n, 12u);
                EXPECT_EQ(product.product.inner, 12 / P);
                ++seen[product.product.i * P + product.product.k];
            }
        for (auto count : seen)
            EXPECT_EQ(count, 1u);
    }
}

INSTANTIATE_TEST_SUITE_P(Workers, RingTest, ::testing::Values(0, 1, 2, 5),
                         ::testing::PrintToStringParamName());

TEST(RingValidationTest, RejectsReplicationAndMalformedMatrices) {
    dmm::SerialExecutor executor;
    dmm::Matrix<float> a(4), b(4);
    EXPECT_THROW(dmm::matmul(a, b, executor, "ring", 4, 2), std::invalid_argument);
    dmm::Matrix<float> mismatched(3);
    EXPECT_THROW(dmm::matmul(a, mismatched, executor, "ring", 4, 1), std::invalid_argument);
}

TEST(RingValidationTest, DirectBuilderRejectsInvalidDimensions) {
    dmm::BspPlan plan(1);
    const auto oversized = std::numeric_limits<std::size_t>::max();
    EXPECT_THROW(dmm::build_ring(plan, {1, 0, 1}), std::invalid_argument);
    EXPECT_THROW(dmm::build_ring(plan, {oversized, 1, 1}), std::invalid_argument);
}

TEST(RingDispatchTest, RotatesDistinctPayloadsThroughBothShifts) {
    dmm::Matrix<float> a(6), b(6), expected(6);
    for (std::size_t i = 0; i < b.values.size(); ++i)
        b.values[i] = float(i + 1);
    const std::size_t selected_rows[] = {2, 5, 4, 1, 0, 3};
    for (std::size_t row = 0; row < a.size; ++row) {
        a.values[row * a.size + selected_rows[row]] = 1;
        std::copy_n(b.values.begin() + selected_rows[row] * b.size, b.size,
                    expected.values.begin() + row * a.size);
    }
    dmm::ThreadExecutor executor(2);
    const auto [product, trace] = dmm::matmul_traced(a, b, executor, "ring", 3);
    EXPECT_EQ(product.values, expected.values);
    const std::size_t origins[2][3] = {{0, 2, 4}, {4, 0, 2}};
    ASSERT_EQ(trace.supersteps.size(), 3u);
    for (std::size_t round = 0; round < 2; ++round) {
        const auto& messages = trace.supersteps[round].messages;
        ASSERT_EQ(messages.size(), 3u);
        bool sent[3] = {};
        for (const auto& message : messages) {
            ASSERT_LT(message.source, 3u);
            EXPECT_FALSE(sent[message.source]);
            sent[message.source] = true;
            EXPECT_EQ(message.destination, (message.source + 1) % 3);
            EXPECT_EQ(message.words, 12u);
            ASSERT_EQ(message.regions.size(), 1u);
            const auto& region = message.regions.front();
            EXPECT_EQ(region.row, origins[round][message.source]);
            EXPECT_EQ(region.col, 0u);
            EXPECT_EQ(region.rows, 2u);
            EXPECT_EQ(region.cols, 6u);
        }
    }
    EXPECT_TRUE(trace.supersteps.back().messages.empty());
}

TEST(RingDispatchTest, NonSquareNodeCountsAreValid) {
    // Unlike Cannon and 2.5D, the ring never requires P = q*q.
    dmm::SerialExecutor executor;
    dmm::Matrix<float> a(6), b(6);
    std::fill(a.values.begin(), a.values.end(), 1);
    std::fill(b.values.begin(), b.values.end(), 1);
    for (std::size_t P : {3, 5, 6})
        EXPECT_NO_THROW(dmm::validate(dmm::matmul(a, b, executor, "ring", P, 1),
                                      constant_reference(6, 6)));
}
} // namespace
