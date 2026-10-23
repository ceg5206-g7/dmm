#include "dmm/validation.hpp"
#include "dmm/algorithms.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <limits>

namespace {
dmm::Matrix<double> constant_reference(std::size_t n, double value) {
    dmm::Matrix<double> m(n);
    std::fill(m.values.begin(), m.values.end(), value);
    return m;
}
} // namespace

TEST(CoverageTest, CannonExecutesEveryTripleOnceAndIndependentMutationsFail) {
    dmm::SerialExecutor executor;
    dmm::Matrix<float> a(17), b(17);
    auto trace = dmm::matmul_traced(a, b, executor).trace;
    EXPECT_NO_THROW(dmm::validate_coverage(trace));
    const auto original = trace;
    auto step = [&]() -> dmm::SuperstepRecord& {
        for (auto& s : trace.supersteps)
            if (!s.products.empty())
                return s;
        throw std::logic_error("test needs an executed product");
    };
    step().products.pop_back();
    EXPECT_THROW(dmm::validate_coverage(trace), std::runtime_error);
    trace = original;
    step().products.push_back(step().products.front());
    EXPECT_THROW(dmm::validate_coverage(trace), std::runtime_error);
    trace = original;
    step().products[0].product.k = 4;
    EXPECT_THROW(dmm::validate_coverage(trace), std::runtime_error);
    trace = original;
    step().products[0].product.inner = 1;
    EXPECT_THROW(dmm::validate_coverage(trace), std::runtime_error);
    trace = original;
    step().flops[step().products[0].node] = 0;
    EXPECT_THROW(dmm::validate_coverage(trace), std::runtime_error);
    trace = original;
    step().products[0].node = trace.nodes;
    EXPECT_THROW(dmm::validate_coverage(trace), std::runtime_error);
    trace = original;
    trace.coverage = {0, 4, 4};
    EXPECT_THROW(dmm::validate_coverage(trace), std::invalid_argument);
}

TEST(CoverageTest, RectangularRingStyleProductsUseSameKernelHookAndValidator) {
    dmm::ThreadExecutor executor(2);
    dmm::BspPlan plan(2);
    std::vector<dmm::MatrixView> a, b, c;
    for (std::size_t id = 0; id < 2; ++id) {
        a.push_back(plan.matrix_view(plan.allocate_buffer(id, 4), 2, 2));
        b.push_back(plan.matrix_view(plan.allocate_buffer(id, 8), 2, 4));
        c.push_back(plan.matrix_view(plan.allocate_buffer(id, 8), 2, 4));
        std::fill(plan.buffer(a[id].buffer).begin(), plan.buffer(a[id].buffer).end(), 1.f);
        std::fill(plan.buffer(b[id].buffer).begin(), plan.buffer(b[id].buffer).end(), 1.f);
    }
    for (std::size_t k = 0; k < 2; ++k) {
        auto step = plan.add_superstep(dmm::Phase::Multiply);
        for (std::size_t id = 0; id < 2; ++id)
            step.add_gemm(a[id], b[id], c[id], {id, k, 0});
    }
    plan.finalize(4, 4, {2, 2, 1});
    plan.execute(executor);
    EXPECT_NO_THROW(dmm::validate_coverage(plan.trace()));
    EXPECT_EQ(dmm::count(plan.trace()).total_flops, 128u);
    for (auto view : c)
        for (auto value : plan.buffer(view.buffer))
            EXPECT_EQ(value, 4);
}

TEST(CoverageTest, RejectedKernelCallDoesNotEnterTheExecutedTrace) {
    dmm::BspPlan plan(1);
    auto a = plan.matrix_view(plan.allocate_buffer(0, 1), 1, 1);
    auto c = plan.matrix_view(plan.allocate_buffer(0, 1), 1, 1);
    auto step = plan.add_superstep(dmm::Phase::Multiply);
    auto bad = a;
    bad.rows = 2;
    EXPECT_THROW(step.add_gemm(bad, a, c, {0, 0, 0}), std::invalid_argument);
    plan.finalize(1, 1, {1, 1, 1});
    dmm::SerialExecutor executor;
    plan.execute(executor);
    EXPECT_TRUE(plan.trace().supersteps[0].products.empty());
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
