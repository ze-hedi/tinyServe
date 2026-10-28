#pragma once

#include "embeddings_mmap.h"
#include "kv_cache.h"
#include "llama_mmap_block.h"
#include "mat.h"
#include "mmap_block_loader.h"
#include "tail_backend.h"
#include "thread_pool.h"
#include "tokenizer.h"

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
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
            std::size_t num_threads = mat_detail::default_thread_count())
        : config_{config},
          pool_{std::max<std::size_t>(1, num_threads)},
          embed_weights_{load_mapped_single_tensor(root_dir, "embed_tokens.json",
                                                   "embed_tokens", mmap_options)},
          embedding_{&embed_weights_},
          final_norm_weights_{load_mapped_single_tensor(root_dir, "final_norm.json",
                                                        "final_norm", mmap_options)},
          final_norm_{&final_norm_weights_},
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
        if (tail_) return next_token_logits_tail(ids32, cache);
        Mat<T> hidden = forward_hidden(to_positions(ids32), &cache);
        cache.advance(ids32.size());
        return last_row_logits(hidden);
    }

    const Tokenizer& tokenizer() const { return tokenizer_; }

    // Weights, for building a TailBackend from the same mappings.
    const MMapLlamaBlockWeights& block_weights(std::size_t layer) const {
        return block_weights_.at(layer);
    }
    const MMapWeight& embed_weights() const { return embed_weights_; }
    const MMapWeight& final_norm_weights() const { return final_norm_weights_; }

    // Run the last layers (and maybe the head) on a TailBackend. Only the cached paths
    // (next_token_logits with a KVCache) use it; the host keeps every weight, so forward() and
    // the cache-less calls still run entirely on the CPU. Pass nullptr to detach. float only.
    //
    // Split of work: decode (one token) runs layers [0, first_layer) here and the rest on the
    // tail. Prefill (several tokens) still runs every layer here, then copies the new K/V rows
    // of the tail layers to the device. The tail layers' K/V rows produced on the device during
    // decode are copied back lazily, only when a later prefill needs them on the host.
    void set_tail(std::unique_ptr<TailBackend> tail) {
        if (tail) {
            if (!std::is_same_v<T, float>) {
                throw std::logic_error("SmolLM3: a tail backend needs float32");
            }
            if (tail->first_layer() >= config_.n_layers) {
                throw std::invalid_argument("SmolLM3: tail starts past the last layer");
            }
        }
        tail_ = std::move(tail);
        tail_device_rows_ = 0;
        tail_host_rows_ = 0;
    }
    bool has_tail() const { return tail_ != nullptr; }
    const TailBackend* tail() const { return tail_.get(); }

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
                          KVCache<T>* cache = nullptr,
                          std::size_t layer_end = std::numeric_limits<std::size_t>::max()) {
        Mat<T> hidden = embedding_.forward(token_ids);
        const std::size_t end = std::min(blocks_.size(), layer_end);
        for (std::size_t i = 0; i < end; ++i) {
            hidden = blocks_[i]->forward(hidden, cache, i);
        }
        return hidden;
    }

    // Cached next-token logits with the last layers on tail_.
    //
    // Row bookkeeping (K/V rows of the tail layers; "device" = the tail's copy):
    //   tail_device_rows_: rows [0, n) are valid on the device
    //   tail_host_rows_:   rows [0, n) are valid in the host KVCache
    // The host cache length (cache.length()) is the truth for the position; truncating it just
    // shortens both counters. Decode writes one device row, prefill writes the same rows on
    // both sides.
    Mat<T> next_token_logits_tail(const std::vector<uint32_t>& ids32, KVCache<T>& cache) {
        if constexpr (std::is_same_v<T, float>) {
            const std::vector<std::size_t> positions = to_positions(ids32);
            const std::size_t first = tail_->first_layer();
            const std::size_t pos0 = cache.length();
            const std::size_t kv_dim = cache.kv_dim();
            const std::size_t d = config_.d_model;

            tail_device_rows_ = std::min(tail_device_rows_, pos0);
            tail_host_rows_ = std::min(tail_host_rows_, pos0);
            if (tail_device_rows_ < pos0) {
                throw std::logic_error(
                    "SmolLM3: the tail's KV cache is behind the host cache (tail attached "
                    "after tokens were cached?)");
            }
            if (cache.capacity() > tail_->kv_capacity()) {
                throw std::length_error("SmolLM3: the tail's KV cache holds " +
                                        std::to_string(tail_->kv_capacity()) +
                                        " tokens, fewer than the context of " +
                                        std::to_string(cache.capacity()));
            }

            if (positions.size() == 1) {
                // Decode: layers [0, first) here, the rest on the tail.
                Mat<T> hidden = forward_hidden(positions, &cache, first);
                std::vector<float> in = row_of(hidden, 0);
                Mat<T> logits(1, 1);
                if (tail_->has_head()) {
                    const std::size_t vocab = embed_weights_.out_features();
                    std::vector<float> out(vocab);
                    tail_->decode_logits(in.data(), pos0, out.data());
                    logits = Mat<T>(1, vocab, std::move(out));
                } else {
                    std::vector<float> out(d);
                    tail_->decode(in.data(), out.data(), pos0);
                    logits = last_row_logits(Mat<T>(1, d, std::move(out)));
                }
                tail_device_rows_ = pos0 + 1;  // the tail wrote row pos0 of every tail layer
                cache.advance(1);
                return logits;
            }

            // Prefill: every layer here. The tail layers attend over [0, pos0) on the host, so
            // first fetch any of those rows that only exist on the device.
            if (tail_host_rows_ < pos0) {
                const std::size_t rows = pos0 - tail_host_rows_;
                for (std::size_t layer = first; layer < config_.n_layers; ++layer) {
                    tail_->download_kv(layer, tail_host_rows_, rows,
                                       cache.k(layer) + tail_host_rows_ * kv_dim,
                                       cache.v(layer) + tail_host_rows_ * kv_dim);
                }
            }
            Mat<T> hidden = forward_hidden(positions, &cache);
            for (std::size_t layer = first; layer < config_.n_layers; ++layer) {
                tail_->upload_kv(layer, pos0, positions.size(), cache.k(layer) + pos0 * kv_dim,
                                 cache.v(layer) + pos0 * kv_dim);
            }
            tail_device_rows_ = tail_host_rows_ = pos0 + positions.size();

            Mat<T> logits(1, 1);
            if (tail_->has_head()) {
                const std::size_t vocab = embed_weights_.out_features();
                std::vector<float> out(vocab);
                tail_->head_logits(row_of(hidden, hidden.rows() - 1).data(), out.data());
                logits = Mat<T>(1, vocab, std::move(out));
            } else {
                logits = last_row_logits(hidden);
            }
            cache.advance(positions.size());
            return logits;
        } else {
            (void)ids32;
            (void)cache;
            throw std::logic_error("SmolLM3: a tail backend needs float32");
        }
    }

    static std::vector<T> row_of(const Mat<T>& m, std::size_t r) {
        std::vector<T> row(m.cols());
        for (std::size_t c = 0; c < m.cols(); ++c) row[c] = m(r, c);
        return row;
    }

    // (rows, d_model) normalized -> (rows, vocab_size)
    Mat<T> lm_head(const Mat<T>& normed) { return normed.matmul(embed_weights_, pool_); }

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

    // Embedding, column-major (d_model, vocab). Tied: also the lm_head weight.
    MMapWeight embed_weights_;
    Embedding_MMap<T> embedding_;

    // Transformer blocks
    std::vector<MMapLlamaBlockWeights> block_weights_;
    std::vector<std::unique_ptr<LlamaBlock_MMap<T>>> blocks_;

    // Final norm
    MMapWeight final_norm_weights_;
    RMSNorm_MMap<T> final_norm_;

    // Tokenizer
    Tokenizer tokenizer_;

    // Optional GPU tail (see set_tail) and the validity of its K/V rows.
    std::unique_ptr<TailBackend> tail_;
    std::size_t tail_device_rows_ = 0;
    std::size_t tail_host_rows_ = 0;
};
