#include "dmm/sharding.hpp"
#include "dmm/cannon.hpp"
#include <gtest/gtest.h>
#include <limits>
#include <random>

TEST(ShardingTest, RejectsMalformedOwnedAndOverflowingLayoutsBeforeCommunication) {
    dmm::BspPlan plan(16);
    std::vector<std::size_t> rows, tiles;
    for (std::size_t node=0; node<16; ++node) {
        rows.push_back(plan.allocate(node,16)); tiles.push_back(plan.allocate(node,16));
    }
    auto foreign=rows; std::swap(foreign[0],foreign[1]);
    EXPECT_THROW(dmm::rows_to_blocks(plan,16,1,foreign,tiles,0),std::invalid_argument);
    auto short_rows=rows; short_rows[0]=plan.allocate(0,1);
    EXPECT_THROW(dmm::rows_to_blocks(plan,16,1,short_rows,tiles,0),std::invalid_argument);
    EXPECT_THROW(dmm::rows_to_blocks(plan,17,1,rows,tiles,0),std::invalid_argument);
    EXPECT_THROW(plan.seal(16,16,{4,4,4}),std::invalid_argument);
    EXPECT_THROW(dmm::padded_size(0,16),std::invalid_argument);
    EXPECT_THROW(dmm::padded_size(17,0),std::invalid_argument);
    EXPECT_THROW(dmm::padded_size(std::numeric_limits<std::size_t>::max(),2),std::invalid_argument);
}

TEST(ShardingTest, ConversionCountsMatchIndependentFourByFourHandCalculation) {
    dmm::BspPlan plan(16);
    std::vector<std::size_t> rows,tiles,output;
    for (std::size_t node=0; node<16; ++node) {
        rows.push_back(plan.allocate(node,16)); tiles.push_back(plan.allocate(node,16)); output.push_back(plan.allocate(node,16));
    }
    dmm::rows_to_blocks(plan,16,1,rows,tiles,0);
    dmm::blocks_to_rows(plan,16,1,tiles,output,0);
    plan.seal(16,16,{4,4,4}); dmm::SerialExecutor executor; plan.execute(executor);
    const auto counts=dmm::count(plan.trace());
    EXPECT_EQ(counts.total_words,384u); EXPECT_EQ(counts.critical_words,24u);
    EXPECT_EQ(counts.messages,96u); EXPECT_EQ(counts.supersteps,4u);
}

TEST(CommonCannonTest, PaddingCroppingAndEndToEndCountsAreRealExecutedWork) {
    dmm::ThreadExecutor executor(2);
    std::mt19937 random(5206);
    std::uniform_real_distribution<float> value(-1, 1);
    for (std::size_t n : {1, 17, 65, 257}) {
        dmm::Matrix<float> a(n), b(n);
        for (auto& x : a.values) x = value(random);
        for (auto& x : b.values) x = value(random);
        const auto result = dmm::Cannon{}.run(a, b, executor);
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
