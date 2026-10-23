#pragma once

#include "mat.h"

#include <cstddef>
#include <stdexcept>
#include <vector>

template <typename T>
class Embedding {
public:
    Embedding(std::size_t vocab_size, std::size_t embed_dim, std::vector<T> weights)
        : vocab_size_{vocab_size}, embed_dim_{embed_dim},
          weight_(vocab_size, embed_dim, std::move(weights)) {}

    std::size_t vocab_size() const { return vocab_size_; }
    std::size_t embed_dim() const { return embed_dim_; }

    // Look up embeddings for a sequence of token IDs.
    // Returns a (seq_len, embed_dim) matrix.
    Mat<T> forward(const std::vector<std::size_t>& token_ids) const {
        if (token_ids.empty()) {
            throw std::invalid_argument("token_ids must not be empty");
        }

        std::size_t seq_len = token_ids.size();
        std::vector<T> data(seq_len * embed_dim_);

        for (std::size_t i = 0; i < seq_len; ++i) {
            if (token_ids[i] >= vocab_size_) {
                throw std::out_of_range("token_id out of vocabulary range");
            }
            for (std::size_t j = 0; j < embed_dim_; ++j) {
                data[i * embed_dim_ + j] = weight_(token_ids[i], j);
            }
        }

        return Mat<T>(seq_len, embed_dim_, std::move(data));
    }

private:
    std::size_t vocab_size_;
    std::size_t embed_dim_;
    Mat<T> weight_; // (vocab_size, embed_dim)
};
