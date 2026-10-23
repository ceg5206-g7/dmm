#include "dmm/redistribution.hpp"
#include "allocation_watch.hpp"
#include <gtest/gtest.h>
#include <limits>

TEST(RedistributionTest, RowRoundTripAndTilesAreCorrectForBaseAndReplicatedLayoutsWithoutAllocation) {
    for (auto [P, c] : {std::pair{16u, 1u}, {64u, 4u}, {256u, 4u}}) {
        dmm::BspPlan plan(P);
        const auto n = dmm::padded_size(17, P), q = dmm::grid_size(P, c), h = n / P, b = n / q;
        const dmm::MatmulConfig config{n, P, c};
        std::vector<dmm::MatrixView> rows(P), tiles(P), output(P);
        auto expected = [](std::size_t row, std::size_t col) {
            return row < 17 && col < 17 ? float(row * 17 + col) : 0.f;
        };
        for (std::size_t node = 0; node < P; ++node) {
            rows[node] = plan.matrix_view(plan.allocate_buffer(node, n * h), h, n);
            if (node < q * q)
                tiles[node] = plan.matrix_view(plan.allocate_buffer(node, b * b), b, b);
            output[node] = plan.matrix_view(plan.allocate_buffer(node, n * h), h, n);
            for (std::size_t r = 0; r < h; ++r)
                for (std::size_t col = 0; col < n; ++col)
                    plan.buffer(rows[node].buffer)[r * n + col] = expected(node * h + r, col);
            std::fill(plan.buffer(output[node].buffer).begin(), plan.buffer(output[node].buffer).end(), -1);
        }
        dmm::add_rows_to_tiles(plan, config, rows, tiles);
        dmm::add_tiles_to_rows(plan, config, tiles, output);
        plan.finalize(17, n, {q, q, q});
        dmm::SerialExecutor executor;
        allocation_test::Watch watch;
        plan.execute(executor);
        auto allocations = watch.finish();
        EXPECT_EQ(allocations, 0u);
        for (std::size_t node = 0; node < P; ++node) {
            for (std::size_t r = 0; r < h; ++r)
                for (std::size_t col = 0; col < n; ++col)
                    ASSERT_EQ(plan.buffer(output[node].buffer)[r * n + col], expected(node * h + r, col));
            if (node < q * q)
                for (std::size_t r = 0; r < b; ++r)
                    for (std::size_t col = 0; col < b; ++col)
                        ASSERT_EQ(plan.buffer(tiles[node].buffer)[r * b + col],
                                  expected(node / q * b + r, node % q * b + col));
        }
        const auto count = dmm::count(plan.trace());
        const auto retained = c == 1 ? P : q;
        EXPECT_EQ(count.total_words, 2 * (P * q - retained) * h * b);
        EXPECT_EQ(count.messages, 2 * (P * q - retained));
        EXPECT_EQ(count.supersteps, 4u);
        EXPECT_EQ(count.total_flops, 0u);
    }
}
TEST(RedistributionTest, RejectsMalformedOwnersShapesAliasingAndOverflowBeforeAddingSteps) {
    dmm::BspPlan plan(16);
    std::vector<dmm::MatrixView> rows, tiles;
    for (std::size_t node = 0; node < 16; ++node) {
        rows.push_back(plan.matrix_view(plan.allocate_buffer(node, 16), 1, 16));
        tiles.push_back(plan.matrix_view(plan.allocate_buffer(node, 16), 4, 4));
    }
    auto foreign = rows;
    std::swap(foreign[0], foreign[1]);
    EXPECT_THROW(dmm::add_rows_to_tiles(plan, {16, 16, 1}, foreign, tiles), std::invalid_argument);
    auto short_rows = rows;
    short_rows[0] = plan.matrix_view(plan.allocate_buffer(0, 1), 1, 1);
    EXPECT_THROW(dmm::add_rows_to_tiles(plan, {16, 16, 1}, short_rows, tiles), std::invalid_argument);
    EXPECT_THROW(dmm::add_rows_to_tiles(plan, {17, 16, 1}, rows, tiles), std::invalid_argument);
    auto alias = tiles;
    alias[0] = plan.matrix_view(rows[0].buffer, 4, 4);
    EXPECT_THROW(dmm::add_rows_to_tiles(plan, {16, 16, 1}, rows, alias), std::invalid_argument);
    EXPECT_THROW(plan.finalize(16, 16, {4, 4, 4}), std::invalid_argument);
    EXPECT_THROW(dmm::padded_size(0, 16), std::invalid_argument);
    EXPECT_THROW(dmm::padded_size(17, 0), std::invalid_argument);
    EXPECT_THROW(dmm::padded_size(std::numeric_limits<std::size_t>::max(), 2), std::invalid_argument);
}
TEST(RedistributionTest, ConversionCountsMatchIndependentFourByFourHandCalculation) {
    dmm::BspPlan plan(16);
    std::vector<dmm::MatrixView> rows, tiles, output;
    for (std::size_t node = 0; node < 16; ++node) {
        rows.push_back(plan.matrix_view(plan.allocate_buffer(node, 16), 1, 16));
        tiles.push_back(plan.matrix_view(plan.allocate_buffer(node, 16), 4, 4));
        output.push_back(plan.matrix_view(plan.allocate_buffer(node, 16), 1, 16));
    }
    dmm::add_rows_to_tiles(plan, {16, 16, 1}, rows, tiles);
    dmm::add_tiles_to_rows(plan, {16, 16, 1}, tiles, output);
    plan.finalize(16, 16, {4, 4, 4});
    dmm::SerialExecutor executor;
    plan.execute(executor);
    auto count = dmm::count(plan.trace());
    EXPECT_EQ(count.total_words, 384u);
    EXPECT_EQ(count.critical_words, 24u);
    EXPECT_EQ(count.messages, 96u);
    EXPECT_EQ(count.supersteps, 4u);
}
