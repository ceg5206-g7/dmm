#include "dmm/bsp.hpp"
#include <gtest/gtest.h>
#include <type_traits>

TEST(NodeMemoryTest, CountsReplacementOverlapCopiesAndTransferWithoutDoubleCounting) {
    static_assert(!std::is_copy_constructible_v<dmm::Buffer>);
    dmm::SerialExecutor executor;
    dmm::Bsp machine(2);
    auto owned = machine.buffers(3);
    owned[0].assign(6, 2);
    EXPECT_EQ(machine.live_words(0), 6u);
    owned[0].assign(6, 3);
    {
        auto temporary = machine.buffer(0, 2);
        auto moved = std::move(temporary);
        EXPECT_EQ(machine.live_words(0), 8u);
    }
    machine.superstep("send", executor, [&](dmm::Node& node) {
        if (node.id() == 0) node.send(1, 0, owned[0]);
    });
    EXPECT_EQ(machine.live_words(0), 6u);
    EXPECT_EQ(machine.live_words(1), 9u); // Old owned 3 and incoming 6 coexist.
    machine.superstep("receive", executor, [&](dmm::Node& node) {
        if (node.id() == 1) owned[1] = node.receive(0, 0, 6);
    });
    EXPECT_EQ(machine.live_words(1), 6u);
    const auto trace = machine.finish();
    EXPECT_EQ(trace.peak_words, (std::vector<std::size_t>{12, 9}));
    EXPECT_EQ(dmm::count(trace).peak_words, 12u); // Maximum node peak, not their sum.
    owned[0].clear(); owned[1].clear();
    EXPECT_EQ(machine.live_words(0), 0u);
    EXPECT_EQ(machine.live_words(1), 0u);
}

TEST(NodeMemoryTest, PerStepPeaksResetWhileTraceAndCountsTakeLifetimeMaximum) {
    dmm::ThreadExecutor executor(2);
    dmm::Bsp machine(2);
    machine.superstep("large", executor, [&](dmm::Node& node) {
        if (node.id() == 0) {
            auto a = node.buffer(10);
            auto b = node.buffer(5);
        }
    });
    machine.superstep("small", executor, [&](dmm::Node& node) {
        auto a = node.buffer(3);
    });
    const auto trace = machine.finish();
    EXPECT_EQ(trace.supersteps[0].peak_words, (std::vector<std::size_t>{15, 0}));
    EXPECT_EQ(trace.supersteps[1].peak_words, (std::vector<std::size_t>{3, 3}));
    EXPECT_EQ(trace.peak_words, (std::vector<std::size_t>{15, 3}));
    EXPECT_EQ(dmm::count(trace).peak_words, 15u);
    EXPECT_EQ(dmm::count(machine.finish()).peak_words, 0u);
}

TEST(NodeMemoryTest, PayloadCanOutliveMachineAndRejectsUnownedAllocations) {
    dmm::Buffer survivor;
    {
        dmm::Bsp machine(1);
        survivor = machine.buffer(0, 4);
        EXPECT_THROW(machine.buffer(1), std::out_of_range);
        EXPECT_THROW(machine.live_words(1), std::out_of_range);
    }
    survivor.assign(5, 2);
    EXPECT_EQ(survivor.size(), 5u);
    survivor.clear();
    dmm::Buffer unowned;
    EXPECT_THROW(unowned.assign(2, 0), std::logic_error);
}
