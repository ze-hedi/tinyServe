#pragma once

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

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
// Each worker is pinned to one logical CPU. The allowed CPUs are ordered to use one SMT sibling
// per physical core before the second siblings; pinning is best-effort and Linux-only.
//
// Contract: run() must be called from a single thread at a time (fork/join,
// not nested). All state below is guarded by mutex_.
class ThreadPool {
public:
    explicit ThreadPool(std::size_t workers)
        : workers_{std::max<std::size_t>(1, workers)} {
        const std::vector<int> cpus = allowed_cpus();
        if (!cpus.empty()) pin_current_thread(cpus.front());
        threads_.reserve(workers_);
        for (std::size_t i = 0; i < workers_; ++i) {
            const int cpu = cpus.empty() ? -1 : cpus[i % cpus.size()];
            threads_.emplace_back([this, cpu] {
                pin_current_thread(cpu);
                worker_loop();
            });
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
    // Logical CPUs this process may run on, with one SMT sibling per core first.
    static std::vector<int> allowed_cpus() {
        std::vector<int> cpus;
#if defined(__linux__)
        cpu_set_t set;
        CPU_ZERO(&set);
        if (sched_getaffinity(0, sizeof(set), &set) == 0) {
            for (int c = 0; c < CPU_SETSIZE; ++c) {
                if (CPU_ISSET(c, &set)) {
                    cpus.push_back(c);
                }
            }
        }
        if (!cpus.empty()) {
            // On typical x86 topology, adjacent logical IDs are SMT siblings. Taking even positions
            // first selects one sibling per core (P-cores first on the target hybrid machine).
            std::vector<int> ordered;
            ordered.reserve(cpus.size());
            for (std::size_t i = 0; i < cpus.size(); i += 2) ordered.push_back(cpus[i]);
            for (std::size_t i = 1; i < cpus.size(); i += 2) ordered.push_back(cpus[i]);
            cpus = std::move(ordered);
        }
#endif
        return cpus;
    }

    // Best-effort: pin the calling thread to a single logical CPU.
    static void pin_current_thread(int cpu) {
#if defined(__linux__)
        if (cpu < 0) {
            return;
        }
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpu, &set);
        (void)pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#else
        (void)cpu;
#endif
    }

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
