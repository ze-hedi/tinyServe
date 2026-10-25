#pragma once

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

// Persistent fork/join worker pool.
//
// The pool owns its worker threads for its entire lifetime (created once, e.g.
// by SmolLM3) so parallel ops never pay thread spawn/join cost per call.
//
// run(ntasks, f) executes f(0), ..., f(ntasks - 1) — each index exactly once,
// on any worker — and blocks until all of them have completed. The calling
// thread waits (it does not execute tasks itself), matching the previous
// spawn-num_threads-and-join semantics.
//
// Contract: run() must be called from a single thread at a time (fork/join,
// not nested). All state below is guarded by mutex_.
class ThreadPool {
public:
    explicit ThreadPool(std::size_t workers)
        : workers_{std::max<std::size_t>(1, workers)} {
        threads_.reserve(workers_);
        for (std::size_t i = 0; i < workers_; ++i) {
            threads_.emplace_back([this] { worker_loop(); });
        }
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        wake_cv_.notify_all();
        for (auto& thread : threads_) {
            thread.join();
        }
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    std::size_t size() const { return workers_; }

    template <typename F>
    void run(std::size_t ntasks, F&& f) {
        if (ntasks == 0) {
            return;
        }
        if (ntasks == 1) {
            f(0);  // inline: cheaper than waking a worker for one task
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            task_ = std::forward<F>(f);
            next_ = 0;
            completed_ = 0;
            total_ = ntasks;
            ++generation_;
        }
        wake_cv_.notify_all();
        std::unique_lock<std::mutex> lock(mutex_);
        done_cv_.wait(lock, [this] { return completed_ == total_; });
    }

private:
    void worker_loop() {
        std::unique_lock<std::mutex> lock(mutex_);
        std::size_t generation = 0;
        for (;;) {
            wake_cv_.wait(lock, [this, &generation] {
                return stop_ || generation_ != generation;
            });
            if (stop_) {
                return;
            }
            generation = generation_;
            while (next_ < total_) {
                const std::size_t index = next_++;
                lock.unlock();
                task_(index);
                lock.lock();
                ++completed_;
                if (completed_ == total_) {
                    done_cv_.notify_all();
                }
            }
        }
    }

    std::size_t workers_;
    std::vector<std::thread> threads_;
    std::mutex mutex_;
    std::condition_variable wake_cv_;
    std::condition_variable done_cv_;
    std::function<void(std::size_t)> task_;
    std::size_t next_ = 0;
    std::size_t completed_ = 0;
    std::size_t total_ = 0;
    std::size_t generation_ = 0;
    bool stop_ = false;
};
