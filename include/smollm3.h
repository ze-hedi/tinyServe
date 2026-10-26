#pragma once

#include "embeddings_mmap.h"
#include "llama_mmap_block.h"
#include "mat.h"
#include "mmap_block_loader.h"
#include "thread_pool.h"

#include <cstddef>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// SmolLM3 model: embedding + transformer blocks + final norm + lm_head.
// Loads all weights from a root directory via mmap.
template <typename T>
class SmolLM3 {
public:
    struct Config {
        std::size_t n_layers = 36;
        std::size_t n_heads = 16;
        std::size_t n_kv_heads = 4;
        std::size_t head_dim = 128;
        std::size_t d_model = 2048;  // n_heads * head_dim
        std::size_t vocab_size = 49152;
    };

    SmolLM3(const std::filesystem::path& root_dir,
            const Config& config = {},
            const MMapLoadOptions& mmap_options = {},
            std::size_t num_threads = mat_detail::hardware_thread_count())
        : config_{config},
          pool_{std::max<std::size_t>(1, num_threads)},
          embed_weights_{load_mapped_single_tensor(root_dir, "embed_tokens.json",
                                                   "embed_tokens", mmap_options)},
          embedding_{&embed_weights_},
          final_norm_weights_{load_mapped_single_tensor(root_dir, "final_norm.json",
                                                        "final_norm", mmap_options)},
          final_norm_{&final_norm_weights_},
          lm_head_weights_{load_mapped_single_tensor(root_dir, "lm_head.json",
                                                      "lm_head", mmap_options)} {

        block_weights_.resize(config_.n_layers);
        blocks_.reserve(config_.n_layers);

        for (std::size_t i = 0; i < config_.n_layers; ++i) {
            const auto block_dir = root_dir / ("bloc_" + std::to_string(i));
            if (!std::filesystem::is_directory(block_dir)) {
                throw std::runtime_error(
                    "SmolLM3: missing block directory '" + block_dir.string() + "'");
            }
            block_weights_[i] = load_mapped_llama_block(block_dir, mmap_options);
            blocks_.push_back(std::make_unique<LlamaBlock_MMap<T>>(
                config_.n_heads, config_.n_kv_heads, config_.head_dim,
                block_weights_[i], pool_));
        }
    }

    const Config& config() const { return config_; }
    std::size_t n_layers() const { return config_.n_layers; }

    // token_ids -> (seq_len, vocab_size) logits
    Mat<T> forward(const std::vector<std::size_t>& token_ids) {
        // Embedding lookup: token_ids -> (seq_len, d_model)
        Mat<T> hidden = embedding_.forward(token_ids);

        // Transformer blocks
        for (std::size_t i = 0; i < blocks_.size(); ++i) {
            hidden = blocks_[i]->forward(hidden);
        }

        // Final RMSNorm
        hidden = final_norm_.forward(hidden, pool_);

        // lm_head projection: (seq_len, d_model) @ (d_model, vocab_size) -> (seq_len, vocab_size)
        return hidden.matmul(lm_head_weights_, pool_);
    }

private:
    Config config_;
    ThreadPool pool_;

    // Embedding
    MMapWeight embed_weights_;
    Embedding_MMap<T> embedding_;

    // Transformer blocks
    std::vector<MMapLlamaBlockWeights> block_weights_;
    std::vector<std::unique_ptr<LlamaBlock_MMap<T>>> blocks_;

    // Final norm + lm_head
    MMapWeight final_norm_weights_;
    RMSNorm_MMap<T> final_norm_;
    MMapWeight lm_head_weights_;
};
