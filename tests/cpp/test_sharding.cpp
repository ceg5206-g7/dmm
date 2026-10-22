#include "dmm/sharding.hpp"
#include "dmm/cannon.hpp"
#include <gtest/gtest.h>
#include <limits>
#include <random>

TEST(ShardingTest, CommonRowsRoundTripThroughTwoAndReplicatedBaseLayouts) {
    dmm::ThreadExecutor executor(2);
    for (const auto [P, c] : std::vector<std::pair<std::size_t, std::size_t>>{
            {1, 1}, {16, 1}, {64, 1}, {256, 1}, {64, 4}, {256, 4}}) {
        dmm::Bsp machine(P);
        dmm::Matrix<float> input(17);
        for (std::size_t i = 0; i < input.values.size(); ++i) input.values[i] = static_cast<float>(i + 1);
        auto rows = dmm::make_row_shards(machine, input);
        EXPECT_EQ(rows.size, dmm::padded_size(17, P));
        const auto blocks = dmm::rows_to_blocks(machine, executor, rows, c, 1);
        const auto output = dmm::blocks_to_rows(machine, executor, blocks, 2);
        EXPECT_EQ(dmm::gather_row_shards(output).values, input.values);
        const auto trace = machine.finish();
        EXPECT_EQ(dmm::count(trace).supersteps, P == 1 ? 0u : 4u);
        for (const auto& step : trace.supersteps)
            for (const auto& message : step.messages) {
                ASSERT_TRUE(message.region.has_value());
                EXPECT_EQ(message.region->rows * message.region->cols, message.words);
                EXPECT_LE(message.region->row + message.region->rows, rows.size);
                EXPECT_LE(message.region->col + message.region->cols, rows.size);
            }
    }
}

TEST(ShardingTest, ConversionCountsMatchIndependentFourByFourHandCalculation) {
    dmm::SerialExecutor executor;
    dmm::Bsp machine(16);
    dmm::Matrix<float> input(16);
    auto rows = dmm::make_row_shards(machine, input);
    auto blocks = dmm::rows_to_blocks(machine, executor, rows, 1, 0);
    auto restored = dmm::blocks_to_rows(machine, executor, blocks, 0);
    const auto counts = dmm::count(machine.finish());
    EXPECT_EQ(counts.total_words, 2u * 16 * 12); // Each row retains one 4-word segment.
    EXPECT_EQ(counts.critical_words, 2u * 12);
    EXPECT_EQ(counts.messages, 2u * 16 * 3);
    EXPECT_EQ(counts.total_flops, 0u);
    EXPECT_EQ(counts.supersteps, 4u);
}

TEST(ShardingTest, RejectsMalformedForeignAndOverflowingLayoutsBeforeCommunication) {
    dmm::SerialExecutor executor;
    dmm::Bsp machine(16), other(16);
    dmm::Matrix<float> input(17);
    auto rows = dmm::make_row_shards(machine, input);
    EXPECT_THROW(dmm::rows_to_blocks(other, executor, rows, 1, 0), std::invalid_argument);
    rows.values[0].clear();
    EXPECT_THROW(dmm::rows_to_blocks(machine, executor, rows, 1, 0), std::invalid_argument);
    EXPECT_THROW(dmm::gather_row_shards(rows), std::invalid_argument);
    EXPECT_TRUE(machine.finish().supersteps.empty());
    EXPECT_THROW(dmm::padded_size(0, 16), std::invalid_argument);
    EXPECT_THROW(dmm::padded_size(17, 0), std::invalid_argument);
    EXPECT_THROW(dmm::padded_size(std::numeric_limits<std::size_t>::max(), 2), std::invalid_argument);
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
