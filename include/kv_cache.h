#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

// Key/value cache for all transformer layers.
// Per layer, K and V are (capacity, kv_dim) row-major buffers, kv_dim = n_kv_heads * head_dim.
// K is stored after RoPE, so cached rows never need to be rotated again.
//
// Layers write rows [length, length + n) during a forward pass; the shared length only moves
// when the caller commits the pass with advance(n). A pass that throws leaves the cache intact.
template <typename T>
class KVCache {
public:
    KVCache(std::size_t n_layers, std::size_t kv_dim, std::size_t capacity)
        : kv_dim_{kv_dim}, capacity_{capacity},
          k_(n_layers, std::vector<T>(kv_dim * capacity, T{0})),
          v_(n_layers, std::vector<T>(kv_dim * capacity, T{0})) {}

    std::size_t n_layers() const { return k_.size(); }
    std::size_t kv_dim() const { return kv_dim_; }
    std::size_t capacity() const { return capacity_; }
    std::size_t length() const { return length_; }

    T* k(std::size_t layer) { return k_.at(layer).data(); }
    T* v(std::size_t layer) { return v_.at(layer).data(); }
    const T* k(std::size_t layer) const { return k_.at(layer).data(); }
    const T* v(std::size_t layer) const { return v_.at(layer).data(); }

    // Commit n freshly written rows.
    void advance(std::size_t n) {
        if (length_ + n > capacity_) {
            throw std::length_error("KVCache: capacity exceeded");
        }
        length_ += n;
    }

    // Drop the rows from position n onwards (keeps the first n); the buffers are reused.
    void truncate(std::size_t n) {
        if (n > length_) {
            throw std::out_of_range("KVCache: cannot truncate beyond the current length");
        }
        length_ = n;
    }

    // Forget everything; the buffers are reused.
    void reset() { length_ = 0; }

private:
    std::size_t kv_dim_;
    std::size_t capacity_;
    std::size_t length_ = 0;
    std::vector<std::vector<T>> k_;
    std::vector<std::vector<T>> v_;
};
