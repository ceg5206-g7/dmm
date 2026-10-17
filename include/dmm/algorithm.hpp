#pragma once

#include <cassert>
#include <cstddef>
#include <concepts>
#include <limits>
#include <stdexcept>
#include <vector>

namespace dmm {

namespace detail {
template <typename T>
std::size_t elements(std::size_t size) {
    if (size == 0 || size > std::numeric_limits<std::size_t>::max() / size / sizeof(T))
        throw std::invalid_argument("matrix size must be positive and fit in memory");
    return size * size;
}
} // namespace detail

// Borrows storage; the source matrix must stay alive and must not reallocate it.
template <typename T>
struct MatrixBlockView {
    const T* data = nullptr;
    std::size_t size = 0;
    std::size_t stride = 0;

    const T& operator()(std::size_t row, std::size_t col) const noexcept {
        assert(row < size && col < size);
        return data[row * stride + col];
    }
};

template <typename T>
struct Matrix {
    explicit Matrix(std::size_t size) : size(size), values(detail::elements<T>(size)) {}
    std::size_t size;
    std::vector<T> values;

    MatrixBlockView<T> block(std::size_t row, std::size_t col, std::size_t block_size) const {
        if (values.size() != detail::elements<T>(size))
            throw std::invalid_argument("matrix storage must match its square size");
        if (block_size == 0 || row >= size || col >= size ||
            block_size > size - row || block_size > size - col)
            throw std::out_of_range("block must be non-empty and fit within the matrix");
        return {values.data() + row * size + col, block_size, size};
    }
};

class Executor;

template <typename T>
concept Algorithm = requires(const T& algorithm, const Matrix<float>& a, const Matrix<float>& b, Executor& executor) {
    { algorithm.matmul(a, b, executor) } -> std::same_as<Matrix<float>>;
};

} // namespace dmm
