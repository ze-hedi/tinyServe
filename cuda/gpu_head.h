#pragma once

#include "cuda_utils.h"
#include "gpu_kernels.h"
#include "mmap_block_loader.h"

#include <cstddef>
#include <stdexcept>
#include <string>

namespace gpu {

// Final RMSNorm + lm_head on the GPU: hidden (d_model) -> logits (vocab).
// The lm_head weight is the tied embedding (column-major (d_model, vocab)), uploaded once at
// construction. Embedding lookup is not done here: it stays on the host, reading the mmap.
class GpuHead {
public:
    // `stream`: run on this stream so the head can follow the blocks without a sync in between
    // (borrowed, must outlive the head). nullptr creates a private stream.
    GpuHead(const MMapWeight& final_norm, const MMapWeight& lm_head, float rms_eps = 1e-6f,
            cudaStream_t stream = nullptr)
        : d_model_{lm_head.in_features()}, vocab_{lm_head.out_features()}, eps_{rms_eps} {
        if (final_norm.in_features() != d_model_ || final_norm.out_features() != 1) {
            throw std::invalid_argument("GpuHead: final_norm must be a (d_model, 1) vector");
        }
        const std::size_t need = d_model_ * vocab_ * sizeof(float);
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        if (need > free_bytes) {
            throw std::runtime_error("GpuHead: lm_head needs " + std::to_string(need >> 20) +
                                     " MiB but only " + std::to_string(free_bytes >> 20) +
                                     " MiB of GPU memory are free");
        }

        if (stream) {
            stream_ = stream;
        } else {
            CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
            owns_stream_ = true;
        }
        gamma_ = DeviceBuffer<float>(d_model_);
        gamma_.upload(final_norm.data(), d_model_);
        weight_ = DeviceBuffer<float>(d_model_ * vocab_);
        weight_.upload(lm_head.data(), d_model_ * vocab_);
        hidden_ = DeviceBuffer<float>(d_model_);
        normed_ = DeviceBuffer<float>(d_model_);
        logits_ = DeviceBuffer<float>(vocab_);
    }

    ~GpuHead() {
        if (owns_stream_ && stream_) {
            cudaStreamSynchronize(stream_);
            cudaStreamDestroy(stream_);
        }
    }
    GpuHead(const GpuHead&) = delete;
    GpuHead& operator=(const GpuHead&) = delete;

    std::size_t d_model() const { return d_model_; }
    std::size_t vocab_size() const { return vocab_; }

    // Host to host: `hidden` (d_model) -> `logits_out` (vocab_size).
    void logits(const float* hidden, float* logits_out) {
        CUDA_CHECK(cudaMemcpyAsync(hidden_.data(), hidden, d_model_ * sizeof(float),
                                   cudaMemcpyHostToDevice, stream_));
        const float* result = logits_device(hidden_.data());
        CUDA_CHECK(cudaMemcpyAsync(logits_out, result, vocab_ * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    // Device in, device out, asynchronous on the stream. The returned pointer (vocab_size
    // floats) is an internal buffer, valid until the next call.
    const float* logits_device(const float* device_hidden) {
        rmsnorm(device_hidden, gamma_.data(), normed_.data(), static_cast<int>(d_model_), eps_,
                stream_);
        gemv(normed_.data(), weight_.data(), nullptr, logits_.data(),
             static_cast<int>(d_model_), static_cast<int>(vocab_), stream_);
        return logits_.data();
    }

    void synchronize() { CUDA_CHECK(cudaStreamSynchronize(stream_)); }

private:
    std::size_t d_model_;
    std::size_t vocab_;
    float eps_;
    cudaStream_t stream_ = nullptr;
    bool owns_stream_ = false;
    DeviceBuffer<float> gamma_, weight_, hidden_, normed_, logits_;
};

}  // namespace gpu
