#pragma once

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

// SmolLM3 model: a stack of LLaMA-style mmap transformer blocks.
// Loads all bloc_0/ .. bloc_{N-1}/ from a root directory and chains forward passes.
// For now this only handles the transformer blocks (no embedding, final norm, or lm_head).
template <typename T>
class SmolLM3 {
public:
    struct Config {
        std::size_t n_layers = 36;
        std::size_t n_heads = 16;
        std::size_t n_kv_heads = 4;
        std::size_t head_dim = 128;
        std::size_t d_model = 2048;  // n_heads * head_dim
    };

    SmolLM3(const std::filesystem::path& root_dir,
            const Config& config = {},
            const MMapLoadOptions& mmap_options = {},
            std::size_t num_threads = mat_detail::hardware_thread_count())
        : config_{config},
          pool_{std::max<std::size_t>(1, num_threads)} {

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

    // x: (seq_len, d_model) -> (seq_len, d_model)
    // Passes input through all transformer blocks sequentially.
    Mat<T> forward(const Mat<T>& x) {
        if (x.cols() != config_.d_model) {
            throw std::invalid_argument("SmolLM3: input cols must equal d_model");
        }
        Mat<T> hidden = blocks_[0]->forward(x);
        for (std::size_t i = 1; i < blocks_.size(); ++i) {
            hidden = blocks_[i]->forward(hidden);
        }
        return hidden;
    }

private:
    Config config_;
    ThreadPool pool_;
    std::vector<MMapLlamaBlockWeights> block_weights_;
    std::vector<std::unique_ptr<LlamaBlock_MMap<T>>> blocks_;
};
