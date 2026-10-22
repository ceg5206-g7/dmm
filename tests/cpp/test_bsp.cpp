#include "dmm/bsp.hpp"
#include <gtest/gtest.h>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
class BspTest : public ::testing::TestWithParam<std::size_t> {};

TEST_P(BspTest, DeliversCopiedPayloadsInTheNextSuperstepAndRecordsThem) {
    auto executor = dmm::make_executor(GetParam());
    dmm::Bsp machine(3);
    std::vector<std::vector<float>> owned{{1}, {2, 3}, {4, 5, 6}};
    std::vector<std::vector<float>> received(3);
    // Node r sends r + 1 words to node r + 1 and does r FLOPs.
    machine.superstep("ring", *executor, [&](dmm::Node& node) {
        const auto r = node.id();
        node.send((r + 1) % 3, 7, owned[r]);
        node.compute(r);
        owned[r].assign(owned[r].size(), 0);
    });
    machine.superstep("drain", *executor, [&](dmm::Node& node) {
        const auto source = (node.id() + 2) % 3;
        received[node.id()] = node.receive(source, 7, source + 1);
    });
    EXPECT_EQ(received, (std::vector<std::vector<float>>{{4, 5, 6}, {1}, {2, 3}}));
    const auto trace = machine.finish();
    ASSERT_EQ(trace.nodes, 3u);
    ASSERT_EQ(trace.supersteps.size(), 2u);
    EXPECT_EQ(trace.supersteps[0].label, "ring");
    EXPECT_EQ(trace.supersteps[0].flops, (std::vector<std::size_t>{0, 1, 2}));
    ASSERT_EQ(trace.supersteps[0].messages.size(), 3u);
    for (std::size_t r = 0; r < 3; ++r) {
        const auto& message = trace.supersteps[0].messages[r];
        EXPECT_EQ(message.source, r);
        EXPECT_EQ(message.destination, (r + 1) % 3);
        EXPECT_EQ(message.tag, 7u);
        EXPECT_EQ(message.words, r + 1);
    }
    EXPECT_TRUE(trace.supersteps[1].messages.empty());
    // h = max(sent, received): node 0 sends 1 and receives 3 words.
    EXPECT_EQ(dmm::count(trace), (dmm::Counts{2, 3, 2, 6, 3, 3}));
    EXPECT_EQ(dmm::count(trace.supersteps[0]), (dmm::Counts{1, 3, 2, 6, 3, 3}));
    EXPECT_EQ(dmm::count(trace.supersteps[1]), (dmm::Counts{1, 0, 0, 0, 0, 0}));
    EXPECT_TRUE(machine.finish().supersteps.empty());
}

TEST_P(BspTest, CountsManyToOneTrafficAtTheReceiver) {
    auto executor = dmm::make_executor(GetParam());
    dmm::Bsp machine(4);
    const std::vector<float> payload(5, 1);
    machine.superstep("gather", *executor, [&](dmm::Node& node) {
        if (node.id() != 0) node.send(0, 0, payload);
    });
    machine.superstep("drain", *executor, [&](dmm::Node& node) {
        if (node.id() == 0)
            for (std::size_t source = 1; source < 4; ++source) node.receive(source, 0, payload.size());
    });
    EXPECT_EQ(dmm::count(machine.finish()), (dmm::Counts{2, 0, 0, 15, 15, 3}));
}

INSTANTIATE_TEST_SUITE_P(Workers, BspTest, ::testing::Values(0, 1, 2, 5),
                         ::testing::PrintToStringParamName());

TEST(BspValidationTest, RejectsInvalidSendsAndReceives) {
    dmm::SerialExecutor executor;
    EXPECT_THROW(dmm::Bsp{0}, std::invalid_argument);
    dmm::Bsp machine(2);
    const std::vector<float> payload{1, 2};
    const auto in_superstep = [&](auto body) {
        machine.superstep("check", executor, [&](dmm::Node& node) {
            if (node.id() == 0) body(node);
        });
    };
    EXPECT_THROW(in_superstep([&](dmm::Node& node) { node.send(2, 0, payload); }), std::out_of_range);
    EXPECT_THROW(in_superstep([&](dmm::Node& node) { node.send(0, 0, payload); }), std::invalid_argument);
    EXPECT_THROW(in_superstep([&](dmm::Node& node) { node.receive(2, 0, 0); }), std::out_of_range);
    EXPECT_THROW(in_superstep([&](dmm::Node& node) { node.receive(1, 0, 0); }), std::logic_error);
    EXPECT_THROW(in_superstep([&](dmm::Node& node) {
        node.send(1, 0, payload);
        node.send(1, 0, payload);
    }), std::logic_error);
}

TEST(BspValidationTest, RejectsWrongLengthsAndUnreceivedMessages) {
    dmm::SerialExecutor executor;
    const std::vector<float> payload{1, 2};
    const auto send = [&](dmm::Node& node) {
        if (node.id() == 0) node.send(1, 3, payload);
    };
    {
        dmm::Bsp machine(2);
        machine.superstep("send", executor, send);
        EXPECT_THROW(machine.superstep("receive", executor, [](dmm::Node& node) {
            if (node.id() == 1) node.receive(0, 3, 1);
        }), std::logic_error);
    }
    {
        dmm::Bsp machine(2);
        machine.superstep("send", executor, send);
        EXPECT_THROW(machine.superstep("ignore", executor, [](dmm::Node&) {}), std::logic_error);
    }
    {
        dmm::Bsp machine(2);
        machine.superstep("send", executor, send);
        EXPECT_THROW(machine.finish(), std::logic_error);
    }
}

TEST(BspCostTest, ChargesCriticalWorkWordsAndSupersteps) {
    const dmm::Counts counts{5, 128, 8, 120, 8, 120};
    EXPECT_DOUBLE_EQ(dmm::bsp_cost(counts, {}), 8);
    EXPECT_DOUBLE_EQ(dmm::bsp_cost(counts, {1, 10, 100}), 8 + 80 + 500);
    EXPECT_DOUBLE_EQ(dmm::bsp_cost(counts, {0.5, 0, 0}), 4);
    EXPECT_THROW(dmm::bsp_cost(counts, {1, -1, 0}), std::invalid_argument);
    EXPECT_THROW(dmm::bsp_cost(counts, {1, 0, std::numeric_limits<double>::infinity()}),
                 std::invalid_argument);
}
} // namespace
