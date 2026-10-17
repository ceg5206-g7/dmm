#include "dmm/cannon.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstddef>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
#include <type_traits>
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

TEST(CannonDispatchTest, SubmitsSixteenTilesInOneBatch) {
    struct RecordingExecutor final : dmm::Executor {
        std::size_t batches = 0;
        std::vector<std::size_t> indices;
        void for_each(std::size_t count, const std::function<void(std::size_t)>& task) override {
            ++batches;
            for (std::size_t i = 0; i < count; ++i) {
                indices.push_back(i);
                task(i);
            }
        }
    } executor;
    dmm::Matrix<float> a(4), b(4);
    std::fill(a.values.begin(), a.values.end(), 1);
    std::fill(b.values.begin(), b.values.end(), 1);
    const dmm::Cannon cannon;
    EXPECT_NO_THROW(dmm::validate(cannon.matmul(a, b, executor), constant_reference(4, 4)));
    EXPECT_EQ(executor.batches, 1u);
    ASSERT_EQ(executor.indices.size(), 16u);
    for (std::size_t i = 0; i < executor.indices.size(); ++i) EXPECT_EQ(executor.indices[i], i);
}

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
