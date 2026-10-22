#include "dmm/cannon.hpp"
#include <gtest/gtest.h>
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
struct MissingMatmul {};
struct ReferenceResult {
    const dmm::Matrix<float>& matmul(const dmm::Matrix<float>&, const dmm::Matrix<float>&, dmm::Executor&) const;
};
struct MutableAlgorithm {
    dmm::Matrix<float> matmul(const dmm::Matrix<float>&, const dmm::Matrix<float>&, dmm::Executor&);
};
static_assert(!dmm::Algorithm<MissingMatmul>);
static_assert(!dmm::Algorithm<ReferenceResult>);
static_assert(!dmm::Algorithm<MutableAlgorithm>);
static_assert(!std::is_polymorphic_v<dmm::Cannon>);


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
    const dmm::Cannon algorithm;
    std::mt19937 random(5206);
    std::uniform_real_distribution<float> value(-1, 1);
    for (std::size_t n : {4, 8, 12, 16, 132}) {
        SCOPED_TRACE(::testing::Message() << "size=" << n);
        dmm::Matrix<float> a(n), b(n);
        for (auto& x : a.values) x = value(random);
        for (auto& x : b.values) x = value(random);
        const auto a_before = a.values, b_before = b.values;
        const auto expected = reference(a, b);
        const auto result = algorithm.matmul(a, b, *executor);
        EXPECT_NO_THROW(dmm::validate(result, expected));
        const auto repeated = algorithm.matmul(a, b, *executor);
        EXPECT_EQ(result.values, repeated.values);
        EXPECT_EQ(a_before, a.values);
        EXPECT_EQ(b_before, b.values);
    }
}

TEST_P(CannonTest, HandlesZeroIdentityOnesAndCancellation) {
    auto executor = dmm::make_executor(GetParam());
    dmm::Matrix<float> a(4), identity(4);
    for (std::size_t i = 0; i < 4; ++i) identity.values[i * 4 + i] = 1;
    const dmm::Cannon cannon;
    EXPECT_NO_THROW(dmm::validate(cannon.matmul(a, identity, *executor), constant_reference(4, 0)));
    std::fill(a.values.begin(), a.values.end(), 1);
    EXPECT_NO_THROW(dmm::validate(cannon.matmul(a, identity, *executor), constant_reference(4, 1)));
    std::fill(identity.values.begin(), identity.values.end(), 1);
    EXPECT_NO_THROW(dmm::validate(cannon.matmul(a, identity, *executor), constant_reference(4, 4)));
    for (std::size_t k = 0; k < 4; ++k)
        for (std::size_t j = 0; j < 4; ++j) identity.values[k * 4 + j] = k % 2 ? -1 : 1;
    EXPECT_NO_THROW(dmm::validate(cannon.matmul(a, identity, *executor), constant_reference(4, 0)));
}

TEST_P(CannonTest, PreservesTileAccumulationOrder) {
    auto executor = dmm::make_executor(GetParam());
    dmm::Matrix<float> a(4), b(4);
    for (std::size_t i = 0; i < 4; ++i) {
        a.values[i * 4] = 1e8f;
        a.values[i * 4 + 1] = 1;
        a.values[i * 4 + 2] = -1e8f;
        a.values[i * 4 + 3] = 1;
    }
    std::fill(b.values.begin(), b.values.end(), 1);
    const dmm::Cannon cannon;
    // The four starting k indices leave different FP32 cancellation residuals.
    const std::vector<float> expected{1, 0, 1, 0, 0, 1, 0, 1,
                                    1, 0, 1, 0, 0, 1, 0, 1};
    EXPECT_EQ(cannon.matmul(a, b, *executor).values, expected);
}

INSTANTIATE_TEST_SUITE_P(Workers, CannonTest, ::testing::Values(0, 1, 2, 5),
                         ::testing::PrintToStringParamName());

TEST(CannonValidationTest, RejectsMismatchedUnsupportedAndMalformedMatrices) {
    dmm::SerialExecutor executor;
    dmm::Matrix<float> a(4), b(3);
    const dmm::Cannon cannon;
    EXPECT_THROW(cannon.matmul(a, b, executor), std::invalid_argument);
    for (std::size_t n : {1, 3, 17, 65}) {
        SCOPED_TRACE(::testing::Message() << "size=" << n);
        dmm::Matrix<float> unsupported(n);
        EXPECT_THROW(cannon.matmul(unsupported, unsupported, executor), std::invalid_argument);
    }
    a.values.pop_back();
    EXPECT_THROW(cannon.matmul(a, a, executor), std::invalid_argument);
}

TEST(ValidationTest, RejectsNonFiniteIncorrectAndMismatchedResults) {
    dmm::Matrix<float> a(4);
    const auto expected = constant_reference(4, 0);
    a.values[0] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_THROW(dmm::validate(a, expected), std::runtime_error);
    a.values[0] = 1;
    EXPECT_THROW(dmm::validate(a, expected), std::runtime_error);
    EXPECT_THROW(dmm::validate(a, dmm::Matrix<double>(3)), std::invalid_argument);
}

TEST(ValidationTest, ChecksBothMatrixShapesAndStorage) {
    dmm::Matrix<float> result(4);
    dmm::Matrix<double> expected(4);
    expected.size = 2;
    EXPECT_THROW(dmm::validate(result, expected), std::invalid_argument);
    expected.size = 4;
    expected.values.pop_back();
    EXPECT_THROW(dmm::validate(result, expected), std::invalid_argument);
    expected.values.push_back(0);
    result.values.pop_back();
    EXPECT_THROW(dmm::validate(result, expected), std::invalid_argument);
    expected.values.pop_back();
    EXPECT_THROW(dmm::validate(result, expected), std::invalid_argument);
}

TEST(ValidationTest, PreservesFP64ReferencePrecision) {
    dmm::Matrix<float> result(1);
    dmm::Matrix<double> expected(1);
    result.values[0] = 1;
    expected.values[0] = 1.0 + std::numeric_limits<double>::epsilon();
    EXPECT_THROW(dmm::validate(result, expected, 0, 0), std::runtime_error);
    const auto error = dmm::validate(result, expected);
    EXPECT_DOUBLE_EQ(error.max_absolute, std::numeric_limits<double>::epsilon());
    expected.values[0] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(dmm::validate(result, expected), std::runtime_error);
}

TEST(CannonDispatchTest, RunsSixteenNodesInEachOfFiveSupersteps) {
    struct RecordingExecutor final : dmm::Executor {
        std::vector<std::size_t> batches;
        void for_each(std::size_t count, const std::function<void(std::size_t)>& task) override {
            batches.push_back(count);
            for (std::size_t i = 0; i < count; ++i) task(i);
        }
    } executor;
    dmm::Matrix<float> a(4), b(4);
    std::fill(a.values.begin(), a.values.end(), 1);
    std::fill(b.values.begin(), b.values.end(), 1);
    const dmm::Cannon cannon;
    EXPECT_NO_THROW(dmm::validate(cannon.matmul(a, b, executor), constant_reference(4, 4)));
    EXPECT_EQ(executor.batches, (std::vector<std::size_t>(5, 16)));
}

class CannonTraceTest : public ::testing::TestWithParam<std::size_t> {};

// Hand count for block size b: the skew moves 12 A and 12 B blocks; each of
// three shifts moves all 32 blocks; every node does 2b^3 FLOPs per round.
TEST_P(CannonTraceTest, MatchesHandComputedCommunication) {
    auto executor = dmm::make_executor(GetParam());
    for (std::size_t n : {4, 8, 12}) {
        SCOPED_TRACE(::testing::Message() << "size=" << n);
        dmm::Matrix<float> a(n), b(n);
        std::fill(a.values.begin(), a.values.end(), 1);
        std::fill(b.values.begin(), b.values.end(), 1);
        const auto [product, trace] = dmm::Cannon{}.run(a, b, *executor);
        EXPECT_NO_THROW(dmm::validate(product, constant_reference(n, static_cast<double>(n))));
        const auto block = n / 4, words = block * block, flops = 2 * words * block;
        ASSERT_EQ(trace.nodes, 16u);
        ASSERT_EQ(trace.supersteps.size(), 5u);
        EXPECT_EQ(dmm::count(trace), (dmm::Counts{5, 16 * 4 * flops, 4 * flops,
                                                  120 * words, 8 * words, 120}));
        EXPECT_EQ(dmm::count(trace.supersteps[0]), (dmm::Counts{1, 0, 0, 24 * words, 2 * words, 24}));
        for (std::size_t s = 1; s < 4; ++s)
            EXPECT_EQ(dmm::count(trace.supersteps[s]),
                      (dmm::Counts{1, 16 * flops, flops, 32 * words, 2 * words, 32}));
        EXPECT_EQ(dmm::count(trace.supersteps[4]), (dmm::Counts{1, 16 * flops, flops, 0, 0, 0}));
        EXPECT_EQ(trace.supersteps[0].label, "skew");
        for (std::size_t s = 1; s < 5; ++s) EXPECT_EQ(trace.supersteps[s].label, "multiply");
    }
}

TEST_P(CannonTraceTest, SkewsAndShiftsAlongGridRowsAndColumns) {
    auto executor = dmm::make_executor(GetParam());
    dmm::Matrix<float> a(8), b(8);
    const auto trace = dmm::Cannon{}.run(a, b, *executor).trace;
    const auto messages_from = [&](std::size_t step, std::size_t source) {
        std::vector<std::pair<std::size_t, std::size_t>> sent; // (destination, tag)
        for (const auto& message : trace.supersteps[step].messages)
            if (message.source == source) {
                EXPECT_EQ(message.words, 4u);
                sent.emplace_back(message.destination, message.tag);
            }
        return sent;
    };
    using Sent = std::vector<std::pair<std::size_t, std::size_t>>;
    // Node (1, 2) = 6 skews A to (1, 1) = 5 and B to (3, 2) = 14, then shifts to (1, 1) and (0, 2).
    EXPECT_EQ(messages_from(0, 6), (Sent{{5, 0}, {14, 1}}));
    EXPECT_EQ(messages_from(1, 6), (Sent{{5, 0}, {2, 1}}));
    // Node (0, 0) keeps both blocks during the skew.
    EXPECT_TRUE(messages_from(0, 0).empty());
    EXPECT_EQ(messages_from(3, 0), (Sent{{3, 0}, {12, 1}}));
    EXPECT_TRUE(messages_from(4, 0).empty());
}

INSTANTIATE_TEST_SUITE_P(Workers, CannonTraceTest, ::testing::Values(0, 2),
                         ::testing::PrintToStringParamName());

TEST(CannonStateTest, ResultsAreIndependentAcrossCallsAndExecutors) {
    dmm::ThreadExecutor executor(2);
    dmm::Matrix<float> a(4), b(4);
    std::fill(a.values.begin(), a.values.end(), 1);
    std::fill(b.values.begin(), b.values.end(), 1);
    const dmm::Cannon algorithm;
    const auto first = algorithm.matmul(a, b, executor);
    std::fill(a.values.begin(), a.values.end(), 2);
    const auto second = algorithm.matmul(a, b, executor);
    dmm::SerialExecutor serial;
    EXPECT_NO_THROW(dmm::validate(algorithm.matmul(a, b, serial), constant_reference(4, 8)));
    EXPECT_NO_THROW(dmm::validate(first, constant_reference(4, 4)));
    EXPECT_NO_THROW(dmm::validate(second, constant_reference(4, 8)));
    bool executed = false;
    EXPECT_NO_THROW(executor.for_each(1, [&](std::size_t) { executed = true; }));
    EXPECT_TRUE(executed);
}
} // namespace
