#include "dmm/redistribution.hpp"

namespace dmm {
std::size_t padded_size(std::size_t original, std::size_t P) {
    if (!original || !P)
        throw std::invalid_argument("matrix size and P must be positive");
    const auto units = (original - 1) / P + 1;
    if (units > std::numeric_limits<std::size_t>::max() / P)
        throw std::invalid_argument("padded size overflows");
    const auto size = units * P;
    detail::elements<float>(size);
    return size;
}
namespace {
void add_conversion(BspPlan& plan, const MatmulConfig& config, std::span<const ConversionMatrix> matrices,
                    std::span<const MatrixView> scratch, bool input) {
    const auto N = config.padded_size, P = config.P, q = grid_size(P, config.c);
    if (matrices.empty() || !N || N % P || plan.nodes() != P || (!scratch.empty() && scratch.size() != P))
        throw std::invalid_argument("invalid row/tile conversion configuration");
    detail::elements<float>(N);
    const auto h = N / P, b = N / q;
    for (const auto& [rows, tiles] : matrices) {
        if (rows.size() != P || tiles.size() != P)
            throw std::invalid_argument("invalid row/tile conversion configuration");
        for (std::size_t node = 0; node < P; ++node) {
            if (plan.owner(rows[node]) != node || rows[node].rows != h || rows[node].cols != N)
                throw std::invalid_argument("row views have wrong shape or owner");
            if (node < q * q &&
                (plan.owner(tiles[node]) != node || tiles[node].rows != b || tiles[node].cols != b))
                throw std::invalid_argument("base tile views have wrong shape or owner");
            if (node < q * q && rows[node].buffer == tiles[node].buffer && node / q == node * h / b && P > 1) {
                if (scratch.empty() || plan.owner(scratch[node]) != node || scratch[node].rows != h ||
                    scratch[node].cols != b || scratch[node].buffer == rows[node].buffer)
                    throw std::invalid_argument("in-place conversion needs separate owned scratch");
            }
        }
        if (P == 1 && (rows[0].buffer != tiles[0].buffer || rows[0].offset != tiles[0].offset ||
                       rows[0].stride != tiles[0].stride))
            throw std::invalid_argument("single-node conversion reuses the same row/tile view");
    }
    if (P == 1)
        return;
    // The rectangle of row shard r in column block j belongs to tile t = i*q + j.
    const auto row_part = [&](const ConversionMatrix& m, std::size_t r, std::size_t j) {
        return m.rows[r].submatrix(0, j * b, h, b);
    };
    const auto tile_part = [&](const ConversionMatrix& m, std::size_t r, std::size_t j) {
        const auto i = r * h / b;
        return m.tiles[i * q + j].submatrix(r * h - i * b, 0, h, b);
    };
    std::vector<PayloadPart> parts;
    std::vector<MatrixView> targets;
    auto send = plan.add_superstep(input ? Phase::InputConversion : Phase::OutputConversion);
    for (std::size_t r = 0; r < P; ++r) {
        const auto i = r * h / b;
        for (std::size_t j = 0; j < q; ++j) {
            const auto t = i * q + j;
            if (r == t)
                continue;
            parts.clear();
            for (const auto& m : matrices)
                parts.push_back({input ? row_part(m, r, j) : tile_part(m, r, j), {r * h, j * b}});
            send.add_send(input ? t : r, parts);
        }
    }
    auto receive = plan.add_superstep(input ? Phase::InputConversion : Phase::OutputConversion);
    // Save each self rectangle before incoming messages overwrite reused row/tile storage.
    for (std::size_t node = 0; node < q * q; ++node)
        if (node / q == node * h / b)
            for (const auto& m : matrices) {
                const auto row = row_part(m, node, node % q), tile = tile_part(m, node, node % q);
                if (row.buffer == tile.buffer) {
                    receive.add_copy(input ? row : tile, scratch[node]);
                    receive.add_copy(scratch[node], input ? tile : row);
                } else
                    receive.add_copy(input ? row : tile, input ? tile : row);
            }
    for (std::size_t r = 0; r < P; ++r) {
        const auto i = r * h / b;
        for (std::size_t j = 0; j < q; ++j) {
            const auto t = i * q + j;
            if (r == t)
                continue;
            targets.clear();
            for (const auto& m : matrices)
                targets.push_back(input ? tile_part(m, r, j) : row_part(m, r, j));
            receive.add_receive(input ? r : t, targets);
        }
    }
}
} // namespace
void add_rows_to_tiles(BspPlan& plan, const MatmulConfig& config, std::span<const ConversionMatrix> matrices,
                       std::span<const MatrixView> scratch) {
    add_conversion(plan, config, matrices, scratch, true);
}
void add_tiles_to_rows(BspPlan& plan, const MatmulConfig& config, std::span<const ConversionMatrix> matrices,
                       std::span<const MatrixView> scratch) {
    add_conversion(plan, config, matrices, scratch, false);
}
void add_rows_to_tiles(BspPlan& plan, const MatmulConfig& config, std::span<const MatrixView> rows,
                       std::span<const MatrixView> tiles, std::span<const MatrixView> scratch) {
    const ConversionMatrix matrix{rows, tiles};
    add_conversion(plan, config, std::span(&matrix, 1), scratch, true);
}
void add_tiles_to_rows(BspPlan& plan, const MatmulConfig& config, std::span<const MatrixView> tiles,
                       std::span<const MatrixView> rows, std::span<const MatrixView> scratch) {
    const ConversionMatrix matrix{rows, tiles};
    add_conversion(plan, config, std::span(&matrix, 1), scratch, false);
}
} // namespace dmm
