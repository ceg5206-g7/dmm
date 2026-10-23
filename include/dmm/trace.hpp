#pragma once
#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace dmm {
enum class Phase { InputConversion, Skew, Replication, Multiply, Reduction, OutputConversion };
inline constexpr std::array<const char*, 6> PHASE_NAMES{"input_conversion", "skew",      "replication",
                                                        "multiply",         "reduction", "output_conversion"};
struct Timing {
    double total = 0, compute = 0, copy = 0, synchronization = 0;
};
struct RunTimings {
    Timing end_to_end;
    std::array<Timing, 6> phases{};
};

struct ElementRegion {
    std::size_t row, col, rows, cols;
};

struct ProductRegion {
    std::size_t i, k, j;     // Uniform block indices, not scalar offsets.
    std::size_t m, n, inner; // Actual GEMM dimensions.
};
struct ProductRecord {
    std::size_t node;
    ProductRegion product;
};
struct CoverageGrid {
    std::size_t row_parts = 0, inner_parts = 0, col_parts = 0;
};

// One logical message; regions follow the payload component order.
struct MessageRecord {
    std::size_t source = 0, destination = 0, words = 0;
    std::vector<ElementRegion> regions;
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

} // namespace dmm
