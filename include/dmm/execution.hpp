#pragma once

#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace dmm {

// Synchronous task batches.
class Executor {
public:
    Executor() = default;
    virtual ~Executor() = default;
    Executor(const Executor&) = delete;
    Executor& operator=(const Executor&) = delete;
    virtual void for_each(std::size_t count, const std::function<void(std::size_t)>& task) = 0;
};

class SerialExecutor final : public Executor {
public:
    void for_each(std::size_t count, const std::function<void(std::size_t)>& task) override;
};

class ThreadExecutor final : public Executor {
public:
    explicit ThreadExecutor(std::size_t workers);
    ~ThreadExecutor() override;
    ThreadExecutor& enter();
    void close();
    void for_each(std::size_t count, const std::function<void(std::size_t)>& task) override;

private:
    void worker(std::size_t index, std::size_t stride);
    void stop();
    std::vector<std::thread> workers_;
    std::mutex batch_mutex_;
    std::mutex mutex_;
    std::condition_variable ready_, done_;
    std::function<void(std::size_t)> task_;
    std::exception_ptr error_;
    std::size_t generation_ = 0, count_ = 0, pending_ = 0;
    bool stopping_ = false;
    bool entered_ = false;
};

// Zero selects serial execution; positive counts select a persistent thread pool.
std::unique_ptr<Executor> make_executor(std::size_t workers);

} // namespace dmm
