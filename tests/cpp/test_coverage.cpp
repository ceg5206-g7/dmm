#include "dmm/coverage.hpp"
#include "dmm/cannon.hpp"
#include <gtest/gtest.h>

TEST(CoverageTest, CannonExecutesEveryTripleOnceAndIndependentMutationsFail) {
    dmm::SerialExecutor executor;
    dmm::Matrix<float> a(17), b(17);
    auto trace = dmm::Cannon{}.run(a, b, executor).trace;
    EXPECT_NO_THROW(dmm::validate_coverage(trace));
    const auto original = trace;
    auto step = [&]() -> dmm::SuperstepRecord& {
        for (auto& s : trace.supersteps) if (!s.products.empty()) return s;
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
    dmm::Bsp machine(2);
    auto a = machine.buffers(4), b = machine.buffers(8), c = machine.buffers(8);
    for (std::size_t id = 0; id < 2; ++id) { a[id].assign(4, 1); b[id].assign(8, 1); }
    for (std::size_t k = 0; k < 2; ++k)
        machine.superstep("multiply", executor, [&](dmm::Node& node) {
            const auto id = node.id();
            node.multiply_add({id, k, 0, 2, 4, 2}, a[id], 2, b[id], 4, c[id], 4);
        });
    auto trace = machine.finish();
    trace.padded_size = 4; trace.coverage = {2, 2, 1};
    EXPECT_NO_THROW(dmm::validate_coverage(trace));
    EXPECT_EQ(dmm::count(trace).total_flops, 128u);
    for (const auto& values : c) for (float value : values) EXPECT_EQ(value, 4);
}

TEST(CoverageTest, FailedKernelCallDoesNotRecordExecutedMultiplication) {
    dmm::SerialExecutor executor;
    dmm::Bsp machine(1);
    auto a = machine.buffer(0, 1), c = machine.buffer(0, 1);
    EXPECT_THROW(machine.superstep("bad", executor, [&](dmm::Node& node) {
        node.multiply_add({0, 0, 0, 2, 2, 2}, a, 2, a, 2, c, 2);
    }), std::invalid_argument);
    machine.superstep("recover", executor, [&](dmm::Node&) {});
    auto trace = machine.finish();
    ASSERT_EQ(trace.supersteps.size(), 1u);
    EXPECT_TRUE(trace.supersteps[0].products.empty());
    EXPECT_EQ(trace.supersteps[0].flops[0], 0u);
}
