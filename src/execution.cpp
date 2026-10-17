#include "dmm/execution.hpp"
#include <stdexcept>

namespace dmm {

void SerialExecutor::for_each(std::size_t count, const std::function<void(std::size_t)>& task) {
    for (std::size_t i = 0; i < count; ++i) task(i);
}

ThreadExecutor::ThreadExecutor(std::size_t workers) {
    if (workers == 0) throw std::invalid_argument("thread executor requires a positive worker count");
    try {
        for (std::size_t i = 0; i < workers; ++i)
            workers_.emplace_back([this, i, workers] { worker(i, workers); });
    } catch (...) {
        stop();
        throw;
    }
}

ThreadExecutor::~ThreadExecutor() { close(); }

ThreadExecutor& ThreadExecutor::enter() {
    std::lock_guard<std::mutex> lock(batch_mutex_);
    if (stopping_) throw std::runtime_error("executor is closed");
    if (entered_) throw std::runtime_error("executor context is already active");
    entered_ = true;
    return *this;
}

void ThreadExecutor::close() {
    std::lock_guard<std::mutex> lock(batch_mutex_);
    if (!stopping_) stop();
}

void ThreadExecutor::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    ready_.notify_all();
    for (auto& thread : workers_) thread.join();
    workers_.clear();
}

void ThreadExecutor::worker(std::size_t index, std::size_t stride) {
    std::size_t observed = 0;
    std::unique_lock<std::mutex> lock(mutex_);
    while (true) {
        ready_.wait(lock, [&] { return stopping_ || generation_ != observed; });
        if (stopping_) return;
        observed = generation_;
        lock.unlock();
        std::exception_ptr error;
        try {
            for (std::size_t i = index; i < count_; i += stride) task_(i);
        } catch (...) {
            error = std::current_exception();
        }
        lock.lock();
        if (error && !error_) error_ = error;
        if (--pending_ == 0) done_.notify_one();
    }
}

void ThreadExecutor::for_each(std::size_t count, const std::function<void(std::size_t)>& task) {
    std::lock_guard<std::mutex> batch_lock(batch_mutex_);
    if (stopping_) throw std::runtime_error("executor is closed");
    std::unique_lock<std::mutex> lock(mutex_);
    task_ = task;
    count_ = count;
    error_ = nullptr;
    pending_ = workers_.size();
    ++generation_;
    ready_.notify_all();
    done_.wait(lock, [&] { return pending_ == 0; });
    task_ = {};
    if (error_) std::rethrow_exception(error_);
}

std::unique_ptr<Executor> make_executor(std::size_t workers) {
    if (workers == 0) return std::make_unique<SerialExecutor>();
    return std::make_unique<ThreadExecutor>(workers);
}

} // namespace dmm
