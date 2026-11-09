#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace dmm {
namespace detail {
struct MemoryLedger {
    std::size_t live = 0, peak = 0, step_peak = 0;
    void acquire(std::size_t words) {
        if (words > std::numeric_limits<std::size_t>::max() - live)
            throw std::overflow_error("node payload memory counter overflow");
        live += words;
        peak = std::max(peak, live);
        step_peak = std::max(step_peak, live);
    }
    void release(std::size_t words) noexcept { live -= words; }
    // Record a momentary peak of live + extra words without retaining them.
    void observe(std::size_t extra) {
        if (extra > std::numeric_limits<std::size_t>::max() - live)
            throw std::overflow_error("node payload memory counter overflow");
        peak = std::max(peak, live + extra);
        step_peak = std::max(step_peak, live + extra);
    }
};
} // namespace detail

// Move-only node-owned payload. Allocated FP32 slots are counted, not container
// metadata. BspPlan moves storage between a node ledger (live payload) and an
// idle ledger (allocated but not live). The ledger outlives its BSP if necessary.
class Buffer {
  public:
    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& other) noexcept : ledger_(std::move(other.ledger_)), values_(std::move(other.values_)) {}
    Buffer& operator=(Buffer&& other) noexcept {
        if (this != &other) {
            release();
            ledger_ = std::move(other.ledger_);
            values_ = std::move(other.values_);
        }
        return *this;
    }
    ~Buffer() { release(); }
    std::size_t size() const noexcept { return values_.size(); }
    bool empty() const noexcept { return values_.empty(); }
    float* data() noexcept { return values_.data(); }
    const float* data() const noexcept { return values_.data(); }
    auto begin() noexcept { return values_.begin(); }
    auto end() noexcept { return values_.end(); }
    auto begin() const noexcept { return values_.begin(); }
    auto end() const noexcept { return values_.end(); }
    float& operator[](std::size_t i) noexcept { return values_[i]; }
    const float& operator[](std::size_t i) const noexcept { return values_[i]; }

  private:
    friend class BspPlan;
    Buffer(std::shared_ptr<detail::MemoryLedger> ledger, std::vector<float> values)
        : ledger_(std::move(ledger)), values_(std::move(values)) {
        ledger_->acquire(values_.capacity());
    }
    void release() noexcept {
        if (ledger_)
            ledger_->release(values_.capacity());
    }
    void rebind(const std::shared_ptr<detail::MemoryLedger>& ledger) {
        ledger->acquire(values_.capacity());
        release();
        ledger_ = ledger;
    }
    std::shared_ptr<detail::MemoryLedger> ledger_;
    std::vector<float> values_;
};

} // namespace dmm
