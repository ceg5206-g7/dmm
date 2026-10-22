#include "dmm/sharding.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace dmm {
namespace {
void validate(std::size_t original, std::size_t n, std::size_t P, const NodeBuffers& values) {
    if (original == 0 || original > n || P == 0 || n % P != 0 || values.size() != P)
        throw std::invalid_argument("invalid shard dimensions or logical-node count");
    detail::elements<float>(n);
}
void validate_rows(const RowShards& rows) {
    validate(rows.original_size, rows.size, rows.values.size(), rows.values);
    for (const auto& value : rows.values)
        if (value.size() != rows.size * (rows.size / rows.values.size()))
            throw std::invalid_argument("row shard payload length does not match its dimensions");
}
void validate_owners(const Bsp& machine, const NodeBuffers& values) {
    if (machine.nodes() != values.size()) throw std::invalid_argument("shards belong to a different BSP size");
    for (std::size_t id = 0; id < values.size(); ++id)
        if (!machine.owns(id, values[id])) throw std::invalid_argument("shard payload belongs to a different node");
}
} // namespace

std::size_t padded_size(std::size_t original, std::size_t P) {
    if (original == 0 || P == 0) throw std::invalid_argument("matrix size and P must be positive");
    const auto units = (original - 1) / P + 1;
    if (units > std::numeric_limits<std::size_t>::max() / P)
        throw std::invalid_argument("padded matrix size overflows");
    const auto n = units * P;
    detail::elements<float>(n);
    return n;
}

RowShards make_row_shards(Bsp& machine, const Matrix<float>& input) {
    if (input.values.size() != detail::elements<float>(input.size))
        throw std::invalid_argument("matrix storage does not match its dimensions");
    const auto n = padded_size(input.size, machine.nodes()), h = n / machine.nodes();
    RowShards rows{input.size, n, machine.buffers(h * n)};
    for (std::size_t i = 0; i < input.size; ++i)
        std::copy_n(input.values.data() + i * input.size, input.size,
                    rows.values[i / h].data() + (i % h) * n);
    return rows;
}

Matrix<float> gather_row_shards(const RowShards& rows) {
    validate_rows(rows);
    Matrix<float> output(rows.original_size);
    const auto h = rows.size / rows.values.size();
    for (std::size_t i = 0; i < rows.original_size; ++i)
        std::copy_n(rows.values[i / h].data() + (i % h) * rows.size, rows.original_size,
                    output.values.data() + i * rows.original_size);
    return output;
}

BlockShards rows_to_blocks(Bsp& machine, Executor& executor, const RowShards& rows,
                           std::size_t c, std::size_t tag) {
    validate_rows(rows);
    validate_owners(machine, rows.values);
    const auto P = machine.nodes(), q = grid_size(P, c), n = rows.size, b = n / q, h = n / P;
    BlockShards output{rows.original_size, n, c, machine.buffers()};
    for (std::size_t id = 0; id < q * q; ++id) output.values[id].assign(b * b, 0);
    if (P == 1) { output.values[0].assign(rows.values[0]); return output; }
    machine.superstep("input_conversion", executor, [&](Node& node) {
        const auto source = node.id(), i = source * h / b, row = source * h - i * b;
        const auto& own = rows.values[source];
        for (std::size_t j = 0; j < q; ++j) {
            const auto destination = i * q + j;
            if (destination == source) {
                for (std::size_t x = 0; x < h; ++x)
                    std::copy_n(own.data() + x * n + j * b, b,
                                output.values[source].data() + (row + x) * b);
            } else {
                auto packed = node.buffer(h * b);
                for (std::size_t x = 0; x < h; ++x)
                    std::copy_n(own.data() + x * n + j * b, b, packed.data() + x * b);
                node.send(destination, tag, packed, ElementRegion{source * h, j * b, h, b});
            }
        }
    });
    machine.superstep("input_conversion", executor, [&](Node& node) {
        const auto id = node.id();
        if (id >= q * q) return;
        const auto i = id / q;
        for (std::size_t source = i * b / h; source < (i + 1) * b / h; ++source) {
            if (source == id) continue;
            const auto incoming = node.receive(source, tag, h * b);
            std::copy(incoming.begin(), incoming.end(), output.values[id].begin() + (source * h - i * b) * b);
        }
    });
    return output;
}

RowShards blocks_to_rows(Bsp& machine, Executor& executor, const BlockShards& blocks, std::size_t tag) {
    const auto P = machine.nodes(), q = grid_size(P, blocks.c), n = blocks.size;
    validate(blocks.original_size, n, P, blocks.values);
    validate_owners(machine, blocks.values);
    const auto b = n / q, h = n / P;
    for (std::size_t id = 0; id < q * q; ++id)
        if (blocks.values[id].size() != b * b) throw std::invalid_argument("base block payload length mismatch");
    RowShards output{blocks.original_size, n, machine.buffers(h * n)};
    if (P == 1) { output.values[0].assign(blocks.values[0]); return output; }
    machine.superstep("output_conversion", executor, [&](Node& node) {
        const auto source = node.id();
        if (source >= q * q) return;
        const auto i = source / q, j = source % q;
        const auto& own = blocks.values[source];
        for (std::size_t destination = i * b / h; destination < (i + 1) * b / h; ++destination) {
            const auto row = destination * h - i * b;
            if (destination == source) {
                for (std::size_t x = 0; x < h; ++x)
                    std::copy_n(own.data() + (row + x) * b, b, output.values[source].data() + x * n + j * b);
            } else {
                auto packed = node.buffer(h * b);
                std::copy_n(own.data() + row * b, h * b, packed.data());
                node.send(destination, tag, packed, ElementRegion{destination * h, j * b, h, b});
            }
        }
    });
    machine.superstep("output_conversion", executor, [&](Node& node) {
        const auto destination = node.id(), i = destination * h / b;
        for (std::size_t j = 0; j < q; ++j) {
            const auto source = i * q + j;
            if (source == destination) continue;
            const auto incoming = node.receive(source, tag, h * b);
            for (std::size_t x = 0; x < h; ++x)
                std::copy_n(incoming.data() + x * b, b, output.values[destination].data() + x * n + j * b);
        }
    });
    return output;
}

} // namespace dmm
