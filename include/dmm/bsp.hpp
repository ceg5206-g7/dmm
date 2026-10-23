#pragma once

#include "dmm/algorithm.hpp"
#include "dmm/execution.hpp"
#include <concepts>
#include <cstddef>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace dmm {

struct ElementRegion {
    std::size_t row, col, rows, cols;
};

struct ProductRegion {
    std::size_t i, k, j; // Uniform block indices, not scalar offsets.
    std::size_t m, n, inner; // Actual GEMM dimensions.
};
struct ProductRecord {
    std::size_t node;
    ProductRegion product;
};
struct CoverageGrid {
    std::size_t row_parts = 0, inner_parts = 0, col_parts = 0;
};

// Payloads are counted in FP32 words; a multiply-add counts as two FLOPs.
struct MessageRecord {
    std::size_t source, destination, tag, words;
    std::optional<ElementRegion> region;
};

struct SuperstepRecord {
    std::string label;
    std::vector<std::size_t> flops; // Indexed by logical node.
    std::vector<MessageRecord> messages;
    std::vector<std::size_t> peak_words;
    std::vector<ProductRecord> products{};
};

struct Trace {
    std::size_t nodes = 0;
    std::vector<SuperstepRecord> supersteps;
    std::vector<std::size_t> peak_words;
    std::size_t original_size = 0, padded_size = 0;
    CoverageGrid coverage{};
};

// h(r, s) is the larger of the words node r sends and receives in superstep s.
// Critical counts sum the per-superstep maxima over nodes, so they add across supersteps.
struct Counts {
    std::size_t supersteps = 0;
    std::size_t total_flops = 0, critical_flops = 0;
    std::size_t total_words = 0, critical_words = 0;
    std::size_t messages = 0;
    std::size_t peak_words = 0;

    Counts& operator+=(const Counts& other) noexcept;
    friend bool operator==(const Counts&, const Counts&) = default;
};

Counts count(const SuperstepRecord& superstep);
Counts count(const Trace& trace);
Counts count_native(const Trace& trace);

// Per-FLOP, per-word and per-superstep costs.
struct CostParameters {
    double w = 1, g = 0, l = 0;
};

// T = w * critical_flops + g * critical_words + l * supersteps.
double bsp_cost(const Counts& counts, const CostParameters& parameters);

struct Traced {
    Matrix<float> product;
    Trace trace;
};

template <typename T>
concept TracedAlgorithm = Algorithm<T> &&
    requires(const T& algorithm, const Matrix<float>& a, const Matrix<float>& b, Executor& executor) {
        { algorithm.run(a, b, executor) } -> std::same_as<Traced>;
    };

} // namespace dmm
