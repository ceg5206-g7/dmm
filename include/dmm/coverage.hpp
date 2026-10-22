#pragma once
#include "dmm/bsp.hpp"

namespace dmm {
// Every uniform block triple must occur exactly once. The grid may be rectangular:
// Cannon/2.5D use q,q,q; a row ring uses P,P,1. Dimensions use padded matrix size.
void validate_coverage(const Trace& trace);
} // namespace dmm
