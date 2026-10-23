#include "allocation_watch.hpp"
#include <atomic>
#include <cstdlib>
#include <new>
namespace {
std::atomic<bool> watching{false};
std::atomic<std::size_t> allocations{0};
} // namespace
void* operator new(std::size_t bytes) {
    if (watching.load(std::memory_order_relaxed))
        ++allocations;
    if (auto* p = std::malloc(bytes ? bytes : 1))
        return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void* operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete[](void* p) noexcept { ::operator delete(p); }
namespace allocation_test {
void start() {
    allocations = 0;
    watching = true;
}
std::size_t stop() {
    watching = false;
    return allocations.load();
}
} // namespace allocation_test
