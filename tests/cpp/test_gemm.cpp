#include "dmm/gemm.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <random>
#include <tuple>
#include <vector>

TEST(GemmTest, RectanglesStridesTailsAndAccumulationMatchIndependentFP64) {
    std::mt19937 random(5206);
    std::uniform_real_distribution<float> sample(-1, 1);
    for (const auto [m, n, k] : std::vector<std::tuple<std::size_t, std::size_t, std::size_t>>{
            {1, 1, 1}, {3, 67, 35}, {33, 31, 65}, {65, 33, 32}}) {
        const auto lda = k + 2, ldb = n + 3, ldc = n + 4;
        std::vector<float> a(m * lda), b(k * ldb), c(m * ldc);
        for (auto& x : a) x = sample(random);
        for (auto& x : b) x = sample(random);
        for (auto& x : c) x = sample(random);
        const auto before_a = a, before_b = b, before_c = c;
        dmm::gemm_add(m, n, k, a, lda, b, ldb, c, ldc);
        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                double expected = before_c[i * ldc + j];
                for (std::size_t x = 0; x < k; ++x)
                    expected += static_cast<double>(a[i * lda + x]) * b[x * ldb + j];
                EXPECT_NEAR(c[i * ldc + j], expected, 1e-4 + 1e-3 * std::abs(expected));
            }
            for (std::size_t j = n; j < ldc; ++j)
                EXPECT_EQ(c[i * ldc + j], before_c[i * ldc + j]);
        }
        EXPECT_EQ(a, before_a);
        EXPECT_EQ(b, before_b);
    }
}

TEST(GemmTest, ZeroInnerDimensionPreservesOutputAndInvalidStorageIsRejected) {
    std::vector<float> c(6, 7), a(6, 1), b(6, 1);
    dmm::gemm_add(2, 3, 0, {}, 0, {}, 3, c, 3);
    EXPECT_EQ(c, std::vector<float>(6, 7));
    EXPECT_THROW(dmm::gemm_add(2, 3, 3, a, 3, b, 3, c, 3), std::invalid_argument);
    EXPECT_THROW(dmm::gemm_add(2, 3, 2, a, 2, b, 2, c, 3), std::invalid_argument);
    EXPECT_THROW(dmm::gemm_add(2, 3, 2, a, 2, b, 3, c, 2), std::invalid_argument);
    EXPECT_THROW(dmm::gemm_add(std::numeric_limits<std::size_t>::max(), 3, 2,
                              a, 2, b, 3, c, 3), std::invalid_argument);
}
