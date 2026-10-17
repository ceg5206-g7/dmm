#include "dmm/algorithm.hpp"
#include <gtest/gtest.h>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace {
static_assert(std::is_trivially_copyable_v<dmm::MatrixBlockView<float>>);
static_assert(std::is_trivially_copyable_v<dmm::MatrixBlockView<double>>);
static_assert(std::is_same_v<decltype(std::declval<dmm::MatrixBlockView<float>>()(0, 0)), const float&>);
static_assert(std::is_same_v<decltype(std::declval<dmm::MatrixBlockView<double>>()(0, 0)), const double&>);

template <typename T>
class MatrixBlockViewTest : public ::testing::Test {};
using ScalarTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(MatrixBlockViewTest, ScalarTypes);

TYPED_TEST(MatrixBlockViewTest, BorrowsStridedStorageWithoutCopying) {
    dmm::Matrix<TypeParam> matrix(6);
    for (std::size_t i = 0; i < matrix.values.size(); ++i) matrix.values[i] = static_cast<TypeParam>(i);
    const auto view = matrix.block(1, 2, 3);
    EXPECT_EQ(view.size, 3u);
    EXPECT_EQ(view.stride, 6u);
    EXPECT_EQ(view.data, matrix.values.data() + 8);
    for (std::size_t i = 0; i < view.size; ++i)
        for (std::size_t j = 0; j < view.size; ++j)
            EXPECT_EQ(&view(i, j), &matrix.values[(i + 1) * matrix.size + j + 2]);
    const auto copy = view;
    matrix.values[15] = -7;
    EXPECT_EQ(view(1, 1), -7);
    EXPECT_EQ(copy(1, 1), -7);
    EXPECT_EQ(matrix.block(4, 4, 2)(1, 1), matrix.values.back());
    EXPECT_EQ(matrix.block(0, 0, 6)(5, 5), matrix.values.back());
}

TYPED_TEST(MatrixBlockViewTest, RejectsInvalidBlocksAndStorage) {
    dmm::Matrix<TypeParam> matrix(6);
    EXPECT_THROW(matrix.block(0, 0, 0), std::out_of_range);
    EXPECT_THROW(matrix.block(6, 0, 1), std::out_of_range);
    EXPECT_THROW(matrix.block(0, 6, 1), std::out_of_range);
    EXPECT_THROW(matrix.block(5, 0, 2), std::out_of_range);
    EXPECT_THROW(matrix.block(0, 5, 2), std::out_of_range);
    EXPECT_THROW(matrix.block(0, 0, std::numeric_limits<std::size_t>::max()), std::out_of_range);
    matrix.values.pop_back();
    EXPECT_THROW(matrix.block(0, 0, 1), std::invalid_argument);
}
} // namespace
