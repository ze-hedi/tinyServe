#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

#define CUDA_CHECK(call)                                                                   \
    do {                                                                                   \
        const cudaError_t cuda_check_err_ = (call);                                        \
        if (cuda_check_err_ != cudaSuccess) {                                              \
            throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" +      \
                                     std::to_string(__LINE__) + " in " #call ": " +        \
                                     cudaGetErrorString(cuda_check_err_));                 \
        }                                                                                  \
    } while (0)

namespace gpu {

// Owning, move-only device allocation of `count` elements of T.
template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t count) : count_{count} {
        if (count_ != 0) CUDA_CHECK(cudaMalloc(&ptr_, count_ * sizeof(T)));
    }
    ~DeviceBuffer() { release(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr_{std::exchange(other.ptr_, nullptr)}, count_{std::exchange(other.count_, 0)} {}
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            release();
            ptr_ = std::exchange(other.ptr_, nullptr);
            count_ = std::exchange(other.count_, 0);
        }
        return *this;
    }

    T* data() { return ptr_; }
    const T* data() const { return ptr_; }
    std::size_t size() const { return count_; }

    // Synchronous copies of `n` elements at element offset `offset`.
    void upload(const T* host, std::size_t n, std::size_t offset = 0) {
        check_range(n, offset);
        CUDA_CHECK(cudaMemcpy(ptr_ + offset, host, n * sizeof(T), cudaMemcpyHostToDevice));
    }
    void download(T* host, std::size_t n, std::size_t offset = 0) const {
        check_range(n, offset);
        CUDA_CHECK(cudaMemcpy(host, ptr_ + offset, n * sizeof(T), cudaMemcpyDeviceToHost));
    }

private:
    void check_range(std::size_t n, std::size_t offset) const {
        if (offset + n > count_) throw std::out_of_range("DeviceBuffer: copy out of range");
    }
    void release() {
        if (ptr_) cudaFree(ptr_);  // no throw in a destructor
        ptr_ = nullptr;
        count_ = 0;
    }

    T* ptr_ = nullptr;
    std::size_t count_ = 0;
};

}  // namespace gpu
