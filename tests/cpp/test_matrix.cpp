#include "dmm/algorithm.hpp"
#include <gtest/gtest.h>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
template <typename T>
class MatrixTest : public ::testing::Test {};
using ScalarTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(MatrixTest, ScalarTypes);

TYPED_TEST(MatrixTest, StoresTypedZeroInitializedValues) {
    dmm::Matrix<TypeParam> matrix(2);
    EXPECT_EQ(matrix.size, 2u);
    EXPECT_EQ(matrix.values, std::vector<TypeParam>(4, 0));
    const TypeParam value = TypeParam{1} + std::numeric_limits<TypeParam>::epsilon();
    matrix.values[0] = value;
    EXPECT_EQ(matrix.values[0], value);
    EXPECT_NE(matrix.values[0], TypeParam{1});
}

TYPED_TEST(MatrixTest, RejectsZeroAndOverflowingSize) {
    EXPECT_THROW(dmm::Matrix<TypeParam>{0}, std::invalid_argument);
    EXPECT_THROW(dmm::Matrix<TypeParam>{std::numeric_limits<std::size_t>::max()}, std::invalid_argument);
}
} // namespace
