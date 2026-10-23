#include "dmm/trace.hpp"
#include <gtest/gtest.h>
#include <limits>
TEST(TraceTest, CostChargesCriticalWorkWordsAndSupersteps) {
    const dmm::Counts counts{5, 128, 8, 120, 8, 120};
    EXPECT_DOUBLE_EQ(dmm::bsp_cost(counts, {}), 8);
    EXPECT_DOUBLE_EQ(dmm::bsp_cost(counts, {1, 10, 100}), 588);
    EXPECT_DOUBLE_EQ(dmm::bsp_cost(counts, {.5, 0, 0}), 4);
    EXPECT_THROW(dmm::bsp_cost(counts, {1, -1, 0}), std::invalid_argument);
    EXPECT_THROW(dmm::bsp_cost(counts, {1, 0, std::numeric_limits<double>::infinity()}),
                 std::invalid_argument);
}
TEST(TraceTest, SumsPerStepMaximaAndSeparatesConversionFromNativeBody) {
    dmm::SuperstepRecord a{"input_conversion", {10, 0}, {{0, 1, 3, {{0, 0, 1, 3}}}}, {12, 9}, {}};
    dmm::SuperstepRecord b{"multiply", {0, 20}, {{1, 0, 5, {{0, 0, 1, 5}}}}, {7, 11}, {}};
    dmm::Trace trace{2, {a, b}, {12, 11}};
    auto all = dmm::count(trace), native = dmm::count_native(trace);
    EXPECT_EQ(all.critical_flops, 30u);
    EXPECT_EQ(all.critical_words, 8u);
    EXPECT_EQ(all.total_words, 8u);
    EXPECT_EQ(all.supersteps, 2u);
    EXPECT_EQ(all.peak_words, 12u);
    EXPECT_EQ(native.critical_flops, 20u);
    EXPECT_EQ(native.critical_words, 5u);
    EXPECT_EQ(native.peak_words, 11u);
    EXPECT_EQ(native.supersteps, 1u);
}
