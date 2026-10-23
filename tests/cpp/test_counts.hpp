#pragma once
#include "dmm/trace.hpp"
#include <gtest/gtest.h>

// Independent schedule checks; payload peaks have separate lifecycle tests.
inline void expect_work_and_communication(const dmm::Counts& actual, const dmm::Counts& expected) {
    EXPECT_EQ(actual.supersteps, expected.supersteps);
    EXPECT_EQ(actual.total_flops, expected.total_flops);
    EXPECT_EQ(actual.critical_flops, expected.critical_flops);
    EXPECT_EQ(actual.total_words, expected.total_words);
    EXPECT_EQ(actual.critical_words, expected.critical_words);
    EXPECT_EQ(actual.messages, expected.messages);
}
