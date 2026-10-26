#pragma once

#include "mat.h"
#include "mmap_block_loader.h"

#include <cstddef>
#include <stdexcept>
#include <vector>

// Embedding layer backed by mmap weights (zero-copy storage).
// Borrows a MMapWeight of shape (vocab_size, embed_dim).
template <typename T>
class Embedding_MMap {
public:
    explicit Embedding_MMap(const MMapWeight* weight)
        : weight_{weight} {}

    std::size_t vocab_size() const { return weight_->rows(); }
    std::size_t embed_dim() const { return weight_->cols(); }

    // Look up embeddings for a sequence of token IDs.
    // Returns a (seq_len, embed_dim) matrix.
    Mat<T> forward(const std::vector<std::size_t>& token_ids) const {
        if (token_ids.empty()) {
            throw std::invalid_argument("token_ids must not be empty");
        }

        const std::size_t seq_len = token_ids.size();
        const std::size_t d = embed_dim();
        const std::size_t v = vocab_size();
        const float* base = weight_->data();

        std::vector<T> data(seq_len * d);
        for (std::size_t i = 0; i < seq_len; ++i) {
            if (token_ids[i] >= v) {
                throw std::out_of_range("token_id out of vocabulary range");
            }
            const float* row = base + token_ids[i] * d;
            for (std::size_t j = 0; j < d; ++j) {
                data[i * d + j] = static_cast<T>(row[j]);
            }
        }

        return Mat<T>(seq_len, d, std::move(data));
    }

private:
    const MMapWeight* weight_;
};
