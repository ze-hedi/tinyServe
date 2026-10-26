#pragma once

#include "embeddings_mmap.h"
#include "kv_cache.h"
#include "llama_mmap_block.h"
#include "mat.h"
#include "mmap_block_loader.h"
#include "thread_pool.h"
#include "tokenizer.h"

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
        float rope_theta = 5000000.0f;
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
                                                      "lm_head", mmap_options)},
          tokenizer_{(root_dir / "tokenizer" / "tokenizer.json").string()} {

        block_weights_.resize(config_.n_layers);
        blocks_.reserve(config_.n_layers);

        for (std::size_t i = 0; i < config_.n_layers; ++i) {
            const auto block_dir = root_dir / ("bloc_" + std::to_string(i));
            if (!std::filesystem::is_directory(block_dir)) {
                throw std::runtime_error(
                    "SmolLM3: missing block directory '" + block_dir.string() + "'");
            }
            block_weights_[i] = load_mapped_llama_block(block_dir, mmap_options);
            bool use_rope = (i % 4 != 3);  // NoPE every 4th layer (3:1 ratio)
            blocks_.push_back(std::make_unique<LlamaBlock_MMap<T>>(
                config_.n_heads, config_.n_kv_heads, config_.head_dim,
                block_weights_[i], pool_, config_.rope_theta, use_rope));
        }
    }

    const Config& config() const { return config_; }
    std::size_t n_layers() const { return config_.n_layers; }

    // token_ids -> (seq_len, vocab_size) logits
    Mat<T> forward(const std::vector<std::size_t>& token_ids) {
        return lm_head(final_norm_.forward(forward_hidden(token_ids), pool_));
    }

    // Cache sized for `capacity` tokens (prompt + everything that will be generated).
    KVCache<T> make_kv_cache(std::size_t capacity) const {
        return KVCache<T>(config_.n_layers, config_.n_kv_heads * config_.head_dim, capacity);
    }

    // String -> tokenize -> forward -> last-position logits (1, vocab_size)
    Mat<T> generate_next(const std::string& text) {
        return next_token_logits(tokenizer_.encode(text));
    }

    // Without cache: token ids -> forward -> last-position logits (1, vocab_size).
    Mat<T> next_token_logits(const std::vector<uint32_t>& ids32) {
        return last_row_logits(forward_hidden(to_positions(ids32)));
    }

    // With cache: ids are the NEW tokens, placed right after what the cache already holds.
    //   prefill: empty cache + whole prompt     decode: one token at a time
    // Returns last-position logits (1, vocab_size) and commits the tokens to the cache.
    Mat<T> next_token_logits(const std::vector<uint32_t>& ids32, KVCache<T>& cache) {
        Mat<T> hidden = forward_hidden(to_positions(ids32), &cache);
        cache.advance(ids32.size());
        return last_row_logits(hidden);
    }

    const Tokenizer& tokenizer() const { return tokenizer_; }

private:
    static std::vector<std::size_t> to_positions(const std::vector<uint32_t>& ids32) {
        if (ids32.empty()) {
            throw std::invalid_argument("SmolLM3: cannot run an empty token sequence");
        }
        return std::vector<std::size_t>(ids32.begin(), ids32.end());
    }

    // Embedding + transformer blocks (before the final norm). With a cache, the tokens sit at
    // positions cache->length() onwards; the caller advances the cache afterwards.
    Mat<T> forward_hidden(const std::vector<std::size_t>& token_ids,
                          KVCache<T>* cache = nullptr) {
        Mat<T> hidden = embedding_.forward(token_ids);
        for (std::size_t i = 0; i < blocks_.size(); ++i) {
            hidden = blocks_[i]->forward(hidden, cache, i);
        }
        return hidden;
    }

    // (rows, d_model) normalized -> (rows, vocab_size)
    Mat<T> lm_head(const Mat<T>& normed) { return normed.matmul(lm_head_weights_, pool_); }

    // Only the last position is needed to pick the next token: norm and project that row alone.
    Mat<T> last_row_logits(const Mat<T>& hidden) {
        const std::size_t d = hidden.cols();
        std::vector<T> row(d);
        for (std::size_t c = 0; c < d; ++c) {
            row[c] = hidden(hidden.rows() - 1, c);
        }
        return lm_head(final_norm_.forward(Mat<T>(1, d, std::move(row)), pool_));
    }

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

    // Tokenizer
    Tokenizer tokenizer_;
};
