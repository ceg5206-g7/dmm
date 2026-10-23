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
    dmm::BspPlan plan(2);
    std::vector<std::size_t> a,b,c;
    for (std::size_t id=0; id<2; ++id) {
        a.push_back(plan.allocate(id,4)); b.push_back(plan.allocate(id,8)); c.push_back(plan.allocate(id,8));
        std::fill(plan.buffer(a[id]).begin(),plan.buffer(a[id]).end(),1.f);
        std::fill(plan.buffer(b[id]).begin(),plan.buffer(b[id]).end(),1.f);
    }
    for (std::size_t k=0; k<2; ++k) {
        auto step=plan.step("multiply");
        for (std::size_t id=0; id<2; ++id)
            plan.multiply_add(step,id,{id,k,0,2,4,2},{a[id],0,2},{b[id],0,4},{c[id],0,4});
    }
    plan.seal(4,4,{2,2,1}); plan.execute(executor);
    EXPECT_NO_THROW(dmm::validate_coverage(plan.trace()));
    EXPECT_EQ(dmm::count(plan.trace()).total_flops,128u);
    for (auto id:c) for (auto value:plan.buffer(id)) EXPECT_EQ(value,4);
}

TEST(CoverageTest, RejectedKernelCallDoesNotEnterTheExecutedTrace) {
    dmm::BspPlan plan(1);
    auto a=plan.allocate(0,1), c=plan.allocate(0,1);
    auto step=plan.step("multiply");
    EXPECT_THROW(plan.multiply_add(step,0,{0,0,0,2,2,2},{a,0,2},{a,0,2},{c,0,2}),std::invalid_argument);
    plan.seal(1,1,{1,1,1}); dmm::SerialExecutor executor; plan.execute(executor);
    EXPECT_TRUE(plan.trace().supersteps[0].products.empty());
}
