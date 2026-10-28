#pragma once

#include "cuda_utils.h"

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace gpu {

// Device-resident K/V rows for the layers that run on the GPU.
// Per local layer, K and V are (capacity, kv_dim) row-major device buffers; K is stored after
// RoPE. There is no length here: the caller owns the position (the host KVCache's length is the
// single source of truth) and passes it to each decode step, so truncating the host cache needs
// no matching call on this side. Rows move between host and device only through
// upload_rows/download_rows.
class GpuKVCache {
public:
    GpuKVCache(std::size_t n_layers, std::size_t kv_dim, std::size_t capacity)
        : kv_dim_{kv_dim}, capacity_{capacity} {
        k_.reserve(n_layers);
        v_.reserve(n_layers);
        for (std::size_t i = 0; i < n_layers; ++i) {
            k_.emplace_back(kv_dim * capacity);
            v_.emplace_back(kv_dim * capacity);
        }
    }

    std::size_t n_layers() const { return k_.size(); }
    std::size_t kv_dim() const { return kv_dim_; }
    std::size_t capacity() const { return capacity_; }

    float* k(std::size_t layer) { return k_.at(layer).data(); }
    float* v(std::size_t layer) { return v_.at(layer).data(); }
    const float* k(std::size_t layer) const { return k_.at(layer).data(); }
    const float* v(std::size_t layer) const { return v_.at(layer).data(); }

    // Rows [row0, row0 + rows) of one layer, host <-> device (host arrays are (rows, kv_dim)).
    void upload_rows(std::size_t layer, std::size_t row0, std::size_t rows, const float* host_k,
                     const float* host_v) {
        k_.at(layer).upload(host_k, rows * kv_dim_, row0 * kv_dim_);
        v_.at(layer).upload(host_v, rows * kv_dim_, row0 * kv_dim_);
    }
    void download_rows(std::size_t layer, std::size_t row0, std::size_t rows, float* host_k,
                       float* host_v) const {
        k_.at(layer).download(host_k, rows * kv_dim_, row0 * kv_dim_);
        v_.at(layer).download(host_v, rows * kv_dim_, row0 * kv_dim_);
    }

private:
    std::size_t kv_dim_;
    std::size_t capacity_;
    std::vector<DeviceBuffer<float>> k_;
    std::vector<DeviceBuffer<float>> v_;
};

}  // namespace gpu
