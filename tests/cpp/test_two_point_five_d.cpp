#include "dmm/algorithms.hpp"
#include "dmm/validation.hpp"
#include <gtest/gtest.h>
#include "test_counts.hpp"
#include <algorithm>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <string>
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

class TwoPointFiveDTest : public ::testing::TestWithParam<std::size_t> {};

TEST_P(TwoPointFiveDTest, MatchesFP64ReferenceAcrossGrids) {
    auto executor = dmm::make_executor(GetParam());
    std::mt19937 random(5206);
    std::uniform_real_distribution<float> value(-1, 1);
    for (const auto [P, c] :
         std::vector<std::pair<std::size_t, std::size_t>>{{1, 1}, {4, 1}, {8, 2}, {32, 2}, {27, 3}, {64, 4}})
        for (std::size_t n : {4, 12, 33}) {
            SCOPED_TRACE(::testing::Message() << "P=" << P << " c=" << c << " size=" << n);
            dmm::Matrix<float> a(n), b(n);
            for (auto& x : a.values)
                x = value(random);
            for (auto& x : b.values)
                x = value(random);
            const auto a_before = a.values, b_before = b.values;
            const auto expected = reference(a, b);
            const auto result = dmm::matmul(a, b, *executor, "2.5d", P, c);
            EXPECT_NO_THROW(dmm::validate(result, expected));
            EXPECT_EQ(a_before, a.values);
            EXPECT_EQ(b_before, b.values);
        }
}

TEST_P(TwoPointFiveDTest, HandlesZeroIdentityOnesAndCancellation) {
    auto executor = dmm::make_executor(GetParam());
    dmm::Matrix<float> a(8), identity(8);
    for (std::size_t i = 0; i < 8; ++i)
        identity.values[i * 8 + i] = 1;
    EXPECT_NO_THROW(dmm::validate(dmm::matmul(a, identity, *executor, "2.5d", 8, 2),
                                  constant_reference(8, 0)));
    std::fill(a.values.begin(), a.values.end(), 1);
    EXPECT_NO_THROW(dmm::validate(dmm::matmul(a, identity, *executor, "2.5d", 8, 2),
                                  constant_reference(8, 1)));
    std::fill(identity.values.begin(), identity.values.end(), 1);
    EXPECT_NO_THROW(dmm::validate(dmm::matmul(a, identity, *executor, "2.5d", 8, 2),
                                  constant_reference(8, 8)));
    for (std::size_t k = 0; k < 8; ++k)
        for (std::size_t j = 0; j < 8; ++j)
            identity.values[k * 8 + j] = k % 2 ? -1 : 1;
    EXPECT_NO_THROW(dmm::validate(dmm::matmul(a, identity, *executor, "2.5d", 8, 2),
                                  constant_reference(8, 0)));
}

// Hand count for (P, c) = (8, 2), n = 8: q = 2, block b = 4, height h = 1,
// one multiply round per layer. Input conversion sends P*q - q = 14 strips of
// h*b words per matrix; replication/reduction move q*q*(c-1) = 4 tiles; the
// skew moves 2*c*(q*q - q) = 8 tiles; the single multiply round never sends.
TEST_P(TwoPointFiveDTest, MatchesHandComputedCommunication) {
    auto executor = dmm::make_executor(GetParam());
    dmm::Matrix<float> a(8), b(8);
    std::fill(a.values.begin(), a.values.end(), 1);
    std::fill(b.values.begin(), b.values.end(), 1);
    auto [product, trace] = dmm::matmul_traced(a, b, *executor, "2.5d", 8, 2);
    EXPECT_NO_THROW(dmm::validate(product, constant_reference(8, 8)));
    const auto words = 16u;                 // b*b
    const auto strip = 4u;                  // h*b
    const auto flops = 2 * words * 4u;      // 2*b^3 per product
    ASSERT_EQ(trace.nodes, 8u);
    const std::vector<dmm::Counts> expected{
        {1, 0, 0, 14 * strip, 4 * strip, 14},  // input conversion A, sends
        {1, 0, 0, 0, 0, 0},                    // input conversion A, receives
        {1, 0, 0, 14 * strip, 4 * strip, 14},  // input conversion B, sends
        {1, 0, 0, 0, 0, 0},                    // input conversion B, receives
        {1, 0, 0, 4 * 2 * words, 2 * words, 4}, // replication, roots send
        {1, 0, 0, 0, 0, 0},                    // replication, leaves receive
        {1, 0, 0, 8 * words, 2 * words, 8},    // skew
        {1, 8 * flops, flops, 0, 0, 0},        // multiply
        {1, 0, 0, 4 * words, words, 4},        // reduction, leaves send
        {1, 4 * words, words, 0, 0, 0},        // reduction, roots receive_sum
        {1, 0, 0, 14 * strip, 4 * strip, 14},  // output conversion, sends
        {1, 0, 0, 0, 0, 0},                    // output conversion, receives
    };
    ASSERT_EQ(trace.supersteps.size(), expected.size());
    for (std::size_t s = 0; s < expected.size(); ++s)
        expect_work_and_communication(dmm::count(trace.supersteps[s]), expected[s]);
    expect_work_and_communication(
        dmm::count(trace), (dmm::Counts{12, 8 * flops + 4 * words, flops + words,
                                        488, 128, 58}));
}

// Hand count for (P, c) = (4, 1), n = 8: q = 2, block b = 4, height h = 2,
// two multiply rounds and no replication or reduction. This must match the
// Cannon schedule on the same grid exactly.
TEST_P(TwoPointFiveDTest, DegenerateReplicationMatchesCannon) {
    auto executor = dmm::make_executor(GetParam());
    dmm::Matrix<float> a(8), b(8);
    std::fill(a.values.begin(), a.values.end(), 1);
    std::fill(b.values.begin(), b.values.end(), 1);
    auto [product, trace] = dmm::matmul_traced(a, b, *executor, "2.5d", 4, 1);
    EXPECT_NO_THROW(dmm::validate(product, constant_reference(8, 8)));
    const auto words = 16u, strip = 8u, flops = 2 * words * 4u;
    ASSERT_EQ(trace.supersteps.size(), 9u);
    const std::vector<dmm::Counts> expected{
        {1, 0, 0, 4 * strip, strip, 4},   // input conversion A, sends
        {1, 0, 0, 0, 0, 0},
        {1, 0, 0, 4 * strip, strip, 4},   // input conversion B, sends
        {1, 0, 0, 0, 0, 0},
        {1, 0, 0, 4 * words, 2 * words, 4},   // skew
        {1, 4 * flops, flops, 8 * words, 2 * words, 8}, // multiply, round 0
        {1, 4 * flops, flops, 0, 0, 0},       // multiply, round 1
        {1, 0, 0, 4 * strip, strip, 4},   // output conversion, sends
        {1, 0, 0, 0, 0, 0},
    };
    for (std::size_t s = 0; s < expected.size(); ++s)
        expect_work_and_communication(dmm::count(trace.supersteps[s]), expected[s]);
    expect_work_and_communication(dmm::count(trace),
                                  (dmm::Counts{9, 8 * flops, 2 * flops, 288, 88, 24}));
    // The c = 1 route must be Cannon's route with identical counts.
    const auto cannon = dmm::matmul_traced(a, b, *executor, "cannon", 4, 1).trace;
    const auto left = dmm::count(trace), right = dmm::count(cannon);
    EXPECT_EQ(left.total_flops, right.total_flops);
    EXPECT_EQ(left.critical_flops, right.critical_flops);
    EXPECT_EQ(left.total_words, right.total_words);
    EXPECT_EQ(left.critical_words, right.critical_words);
    EXPECT_EQ(left.messages, right.messages);
    EXPECT_EQ(left.supersteps, right.supersteps);
}

TEST_P(TwoPointFiveDTest, CoversEveryProductTripleOnce) {
    auto executor = dmm::make_executor(GetParam());
    for (const auto [P, c, q] : std::vector<std::tuple<std::size_t, std::size_t, std::size_t>>{
             {8, 2, 2}, {32, 2, 4}, {27, 3, 3}, {64, 4, 4}}) {
        SCOPED_TRACE(::testing::Message() << "P=" << P << " c=" << c);
        const auto n = 2 * P, block = n / q;
        dmm::Matrix<float> a(n), b(n);
        std::fill(a.values.begin(), a.values.end(), 1);
        std::fill(b.values.begin(), b.values.end(), 1);
        const auto trace = dmm::matmul_traced(a, b, *executor, "2.5d", P, c).trace;
        std::vector<std::size_t> seen(q * q * q);
        for (const auto& step : trace.supersteps)
            for (const auto& product : step.products) {
                ASSERT_LT(product.product.i, q);
                ASSERT_LT(product.product.k, q);
                ASSERT_LT(product.product.j, q);
                EXPECT_EQ(product.product.m, block);
                EXPECT_EQ(product.product.n, block);
                EXPECT_EQ(product.product.inner, block);
                ++seen[(product.product.i * q + product.product.k) * q + product.product.j];
            }
        for (auto count : seen)
            EXPECT_EQ(count, 1u);
    }
}

TEST_P(TwoPointFiveDTest, UsesConversionReplicationAndReductionPhases) {
    auto executor = dmm::make_executor(GetParam());
    dmm::Matrix<float> a(8), b(8);
    std::fill(a.values.begin(), a.values.end(), 1);
    std::fill(b.values.begin(), b.values.end(), 1);
    const auto labels = [&](std::size_t P, std::size_t c) {
        std::vector<std::string> result;
        for (const auto& step : dmm::matmul_traced(a, b, *executor, "2.5d", P, c).trace.supersteps)
            result.push_back(step.label);
        return result;
    };
    EXPECT_EQ(labels(8, 2),
              (std::vector<std::string>{"input_conversion", "input_conversion", "input_conversion",
                                        "input_conversion", "replication", "replication", "skew",
                                        "multiply", "reduction", "reduction", "output_conversion",
                                        "output_conversion"}));
    EXPECT_EQ(labels(4, 1),
              (std::vector<std::string>{"input_conversion", "input_conversion", "input_conversion",
                                        "input_conversion", "skew", "multiply", "multiply",
                                        "output_conversion", "output_conversion"}));
    EXPECT_EQ(labels(1, 1), (std::vector<std::string>{"skew", "multiply"}));
}

INSTANTIATE_TEST_SUITE_P(Workers, TwoPointFiveDTest, ::testing::Values(0, 2),
                         ::testing::PrintToStringParamName());

TEST(TwoPointFiveDValidationTest, RejectsInvalidGridsAndMatrices) {
    dmm::SerialExecutor executor;
    dmm::Matrix<float> a(4), b(4);
    for (const auto [P, c] :
         std::vector<std::pair<std::size_t, std::size_t>>{{16, 2}, {64, 2}, {8, 3}, {4, 0}})
        EXPECT_THROW(dmm::matmul(a, b, executor, "2.5d", P, c), std::invalid_argument);
    dmm::Matrix<float> mismatched(3);
    EXPECT_THROW(dmm::matmul(a, mismatched, executor, "2.5d", 8, 2), std::invalid_argument);
}
} // namespace
