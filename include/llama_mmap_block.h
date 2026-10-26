#pragma once

#include "kv_cache.h"
#include "mat.h"
#include "mmap_block_loader.h"
#include "thread_pool.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

// Apply Rotary Position Embeddings (RoPE) in-place to x: (seq_len, head_dim).
// (same as llama_transformer.h — duplicated to keep this header self-contained)
namespace gqa_mmap_detail {
template <typename T>
void apply_rope(Mat<T>& x, float rope_theta, std::size_t pos_offset = 0) {
    std::size_t seq_len = x.rows();
    std::size_t head_dim = x.cols();
    if (head_dim % 2 != 0) {
        throw std::invalid_argument("apply_rope: head_dim must be even");
    }
    for (std::size_t i = 0; i < seq_len; ++i) {
        std::size_t pos = pos_offset + i;
        for (std::size_t j = 0; j < head_dim / 2; ++j) {
            const float theta = static_cast<float>(pos) /
                std::pow(rope_theta,
                         static_cast<float>(2 * j) / static_cast<float>(head_dim));
            const float cos_t = std::cos(theta);
            const float sin_t = std::sin(theta);
            const float x0 = static_cast<float>(x(i, 2 * j));
            const float x1 = static_cast<float>(x(i, 2 * j + 1));
            x(i, 2 * j) = static_cast<T>(x0 * cos_t - x1 * sin_t);
            x(i, 2 * j + 1) = static_cast<T>(x0 * sin_t + x1 * cos_t);
        }
    }
}
template <typename T>
void apply_rope_heads(Mat<T>& x, std::size_t n_heads, std::size_t head_dim,
                      float rope_theta, std::size_t pos_offset = 0) {
    if (head_dim % 2 != 0) {
        throw std::invalid_argument("apply_rope_heads: head_dim must be even");
    }
    if (x.cols() != n_heads * head_dim) {
        throw std::invalid_argument("apply_rope_heads: invalid matrix dimensions");
    }
    const std::size_t seq_len = x.rows();
    const std::size_t pairs = head_dim / 2;
    std::vector<float> cosines(seq_len * pairs);
    std::vector<float> sines(seq_len * pairs);
    for (std::size_t j = 0; j < pairs; ++j) {
        const float frequency = std::pow(
            rope_theta, static_cast<float>(2 * j) / static_cast<float>(head_dim));
        for (std::size_t i = 0; i < seq_len; ++i) {
            const float theta = static_cast<float>(pos_offset + i) / frequency;
            cosines[i * pairs + j] = std::cos(theta);
            sines[i * pairs + j] = std::sin(theta);
        }
    }

    for (std::size_t h = 0; h < n_heads; ++h) {
        const std::size_t offset = h * head_dim;
        for (std::size_t i = 0; i < seq_len; ++i) {
            for (std::size_t j = 0; j < pairs; ++j) {
                const float cos_t = cosines[i * pairs + j];
                const float sin_t = sines[i * pairs + j];
                const float x0 = static_cast<float>(x(i, offset + 2 * j));
                const float x1 = static_cast<float>(x(i, offset + 2 * j + 1));
                x(i, offset + 2 * j) = static_cast<T>(x0 * cos_t - x1 * sin_t);
                x(i, offset + 2 * j + 1) = static_cast<T>(x0 * sin_t + x1 * cos_t);
            }
        }
    }
}
} // namespace gqa_mmap_detail

// Grouped Query Attention backed by mmap weights (zero-copy).
// Borrows MMapWeight pointers — the caller must keep them alive.
template <typename T>
class GQA_MMap {
public:
    // W_q: (d_model, n_heads * head_dim)
    // W_k: (d_model, n_kv_heads * head_dim)
    // W_v: (d_model, n_kv_heads * head_dim)
    // W_o: (n_heads * head_dim, d_model)
    GQA_MMap(std::size_t n_heads, std::size_t n_kv_heads, std::size_t head_dim,
             const MMapWeight* W_q, const MMapWeight* W_k,
             const MMapWeight* W_v, const MMapWeight* W_o,
             float rope_theta = 5000000.0f, bool use_rope = true)
        : n_heads_{n_heads}, n_kv_heads_{n_kv_heads}, head_dim_{head_dim},
          rope_theta_{rope_theta}, use_rope_{use_rope},
          W_q_{W_q}, W_k_{W_k}, W_v_{W_v}, W_o_{W_o} {
        if (n_heads_ % n_kv_heads_ != 0) {
            throw std::invalid_argument("n_heads must be divisible by n_kv_heads");
        }
        std::size_t d = W_q_->rows();
        if (W_q_->cols() != n_heads_ * head_dim_) {
            throw std::invalid_argument("W_q cols must equal n_heads * head_dim");
        }
        if (W_k_->rows() != d || W_k_->cols() != n_kv_heads_ * head_dim_) {
            throw std::invalid_argument("W_k dimensions mismatch");
        }
        if (W_v_->rows() != d || W_v_->cols() != n_kv_heads_ * head_dim_) {
            throw std::invalid_argument("W_v dimensions mismatch");
        }
        if (W_o_->rows() != n_heads_ * head_dim_ || W_o_->cols() != d) {
            throw std::invalid_argument("W_o dimensions mismatch");
        }
    }

    std::size_t n_heads() const { return n_heads_; }
    std::size_t n_kv_heads() const { return n_kv_heads_; }
    std::size_t head_dim() const { return head_dim_; }
    std::size_t d_model() const { return W_q_->rows(); }

    // x: (seq_len, d_model) -> output: (seq_len, d_model)
    //
    // Without a cache: plain causal attention over x.
    // With a cache: x holds the new tokens, at absolute positions cache->length() onwards.
    // Their K/V rows are appended to the cache for `layer` and attention runs over the whole
    // cached prefix. The caller commits the pass with cache->advance(seq_len).
    Mat<T> forward(const Mat<T>& x, ThreadPool& pool,
                   KVCache<T>* cache = nullptr, std::size_t layer = 0) const {
        if (x.cols() != d_model()) {
            throw std::invalid_argument("input cols must equal d_model");
        }

        const std::size_t seq_len = x.rows();
        const std::size_t pos0 = cache ? cache->length() : 0;
        if (cache) {
            if (cache->kv_dim() != n_kv_heads_ * head_dim_) {
                throw std::invalid_argument("GQA_MMap: KVCache kv_dim mismatch");
            }
            if (pos0 + seq_len > cache->capacity()) {
                throw std::length_error("GQA_MMap: KVCache capacity exceeded");
            }
        }

        // Project using mmap weights
        Mat<T> Q = x.matmul(*W_q_, pool); // (seq_len, n_heads * head_dim)
        Mat<T> K = x.matmul(*W_k_, pool); // (seq_len, n_kv_heads * head_dim)
        Mat<T> V = x.matmul(*W_v_, pool); // (seq_len, n_kv_heads * head_dim)

        if (use_rope_) {
            gqa_mmap_detail::apply_rope_heads(Q, n_heads_, head_dim_, rope_theta_, pos0);
            gqa_mmap_detail::apply_rope_heads(K, n_kv_heads_, head_dim_, rope_theta_, pos0);
        }

        if (cache) {
            const std::size_t kv_dim = n_kv_heads_ * head_dim_;
            T* k_dst = cache->k(layer) + pos0 * kv_dim;
            T* v_dst = cache->v(layer) + pos0 * kv_dim;
            for (std::size_t i = 0; i < seq_len; ++i) {
                for (std::size_t c = 0; c < kv_dim; ++c) {
                    k_dst[i * kv_dim + c] = K(i, c);
                    v_dst[i * kv_dim + c] = V(i, c);
                }
            }
        }

        // Nothing cached before these tokens: K/V are exactly the local ones, use the
        // matmul-based path (fast for long prompts). Otherwise attend over the cache.
        Mat<T> concat = pos0 == 0 ? attend_local(Q, K, V, pool)
                                  : attend_cached(Q, *cache, layer, pos0, pool);

        // Output projection
        return concat.matmul(*W_o_, pool);
    }

private:
    // Causal attention of Q over its own K/V. -> (seq_len, n_heads * head_dim)
    Mat<T> attend_local(const Mat<T>& Q, const Mat<T>& K, const Mat<T>& V,
                        ThreadPool& pool) const {
        const std::size_t n_groups = n_heads_ / n_kv_heads_;
        T scale = T{1} / std::sqrt(static_cast<T>(head_dim_));

        // Output buffer: (seq_len, n_heads * head_dim)
        Mat<T> concat(Q.rows(), n_heads_ * head_dim_);

        // Per-head attention
        for (std::size_t h = 0; h < n_heads_; ++h) {
            std::size_t kv_h = h / n_groups;

            Mat<T> Q_h = Q.slice_cols(h * head_dim_, head_dim_);
            Mat<T> K_h = K.slice_cols(kv_h * head_dim_, head_dim_);
            Mat<T> V_h = V.slice_cols(kv_h * head_dim_, head_dim_);

            // scores = Q_h @ K_h^T / sqrt(head_dim) -> (seq_len, seq_len)
            Mat<T> scores = Q_h.matmul(K_h.transpose(), pool);
            scores.scale(scale);

            scores.masked_fill(-std::numeric_limits<T>::infinity(), true);
            scores.softmax_rows();

            Mat<T> attn_out = scores.matmul(V_h, pool);
            concat.set_cols(h * head_dim_, attn_out);
        }
        return concat;
    }

    // Attention of the new queries (absolute positions pos0 + i) over cache rows [0, pos0 + i].
    // Reads the cache buffers directly: for a single decode row there is nothing worth tiling,
    // and slicing the whole cache per head would copy it on every generated token.
    Mat<T> attend_cached(const Mat<T>& Q, const KVCache<T>& cache, std::size_t layer,
                         std::size_t pos0, ThreadPool& pool) const {
        const std::size_t seq_len = Q.rows();
        const std::size_t n_groups = n_heads_ / n_kv_heads_;
        const std::size_t kv_dim = n_kv_heads_ * head_dim_;
        const T scale = T{1} / std::sqrt(static_cast<T>(head_dim_));
        const T* k_cache = cache.k(layer);
        const T* v_cache = cache.v(layer);

        Mat<T> concat(seq_len, n_heads_ * head_dim_);

        // Heads write disjoint column ranges of concat, so they can run in parallel.
        auto run_heads = [&](std::size_t h_begin, std::size_t h_end) {
            std::vector<T> scores(pos0 + seq_len);
            for (std::size_t h = h_begin; h < h_end; ++h) {
                const std::size_t kv_off = (h / n_groups) * head_dim_;
                for (std::size_t i = 0; i < seq_len; ++i) {
                    const std::size_t visible = pos0 + i + 1;  // causal: keys 0..pos0+i

                    T max_score = -std::numeric_limits<T>::infinity();
                    for (std::size_t c = 0; c < visible; ++c) {
                        const T* k_row = k_cache + c * kv_dim + kv_off;
                        T dot{0};
                        for (std::size_t d = 0; d < head_dim_; ++d) {
                            dot += Q(i, h * head_dim_ + d) * k_row[d];
                        }
                        scores[c] = dot * scale;
                        max_score = std::max(max_score, scores[c]);
                    }

                    T sum{0};
                    for (std::size_t c = 0; c < visible; ++c) {
                        scores[c] = std::exp(scores[c] - max_score);
                        sum += scores[c];
                    }

                    for (std::size_t d = 0; d < head_dim_; ++d) {
                        T acc{0};
                        for (std::size_t c = 0; c < visible; ++c) {
                            acc += scores[c] * v_cache[c * kv_dim + kv_off + d];
                        }
                        concat(i, h * head_dim_ + d) = acc / sum;
                    }
                }
            }
        };

        const std::size_t worker_count = std::min(pool.size(), n_heads_);
        pool.run(worker_count, [&](std::size_t t) {
            run_heads(n_heads_ * t / worker_count, n_heads_ * (t + 1) / worker_count);
        });
        return concat;
    }

    std::size_t n_heads_;
    std::size_t n_kv_heads_;
    std::size_t head_dim_;
    float rope_theta_;
    bool use_rope_;
    const MMapWeight* W_q_;
    const MMapWeight* W_k_;
    const MMapWeight* W_v_;
    const MMapWeight* W_o_;
};

// SwiGLU MLP backed by mmap weights (zero-copy).
// out = (silu(x @ W_gate) ⊙ (x @ W_up)) @ W_down
template <typename T>
class SwiGLU_MMap {
public:
    // W_gate: (d_model, d_ff)
    // W_up:   (d_model, d_ff)
    // W_down: (d_ff, d_model)
    SwiGLU_MMap(const MMapWeight* W_gate, const MMapWeight* W_up, const MMapWeight* W_down)
        : W_gate_{W_gate}, W_up_{W_up}, W_down_{W_down} {
        std::size_t d = W_gate_->rows();
        std::size_t ff = W_gate_->cols();
        if (W_up_->rows() != d || W_up_->cols() != ff) {
            throw std::invalid_argument("SwiGLU_MMap: W_up dimensions must match W_gate");
        }
        if (W_down_->rows() != ff || W_down_->cols() != d) {
            throw std::invalid_argument("SwiGLU_MMap: W_down must be (d_ff, d_model)");
        }
    }

    std::size_t d_model() const { return W_gate_->rows(); }
    std::size_t d_ff() const { return W_gate_->cols(); }

    // x: (seq_len, d_model) -> (seq_len, d_model)
    Mat<T> forward(const Mat<T>& x, ThreadPool& pool) const {
        if (x.cols() != d_model()) {
            throw std::invalid_argument("SwiGLU_MMap: input cols must equal d_model");
        }
        Mat<T> gate = x.matmul(*W_gate_, pool);  // (seq_len, d_ff)
        Mat<T> up = x.matmul(*W_up_, pool);      // (seq_len, d_ff)

        gate.silu_hadamard(up, pool);
        return gate.matmul(*W_down_, pool);  // (seq_len, d_model)
    }

private:
    const MMapWeight* W_gate_;
    const MMapWeight* W_up_;
    const MMapWeight* W_down_;
};

// RMSNorm backed by mmap gamma weights (zero-copy).
// gamma is a (1, d_model) row vector stored as an MMapWeight.
template <typename T>
class RMSNorm_MMap {
public:
    RMSNorm_MMap(const MMapWeight* gamma, T eps = static_cast<T>(1e-6f))
        : gamma_{gamma}, eps_{eps} {
        if (gamma_->rows() != 1) {
            throw std::invalid_argument("RMSNorm_MMap: gamma must be a (1, d_model) row vector");
        }
    }

    std::size_t d_model() const { return gamma_->cols(); }

    // x: (seq_len, d_model) -> (seq_len, d_model)
    Mat<T> forward(const Mat<T>& x, ThreadPool& pool) const {
        if (x.cols() != d_model()) {
            throw std::invalid_argument("RMSNorm_MMap: input cols must equal d_model");
        }
        std::size_t seq_len = x.rows();
        std::size_t d = d_model();
        const float* g = gamma_->data();
        Mat<T> result(seq_len, d);
        if (seq_len == 0) {
            return result;
        }

        auto worker = [&](std::size_t r_start, std::size_t r_end) {
            for (std::size_t r = r_start; r < r_end; ++r) {
                T sum_sq{0};
                for (std::size_t c = 0; c < d; ++c) {
                    sum_sq += x(r, c) * x(r, c);
                }
                T rms = std::sqrt(sum_sq / static_cast<T>(d) + eps_);
                for (std::size_t c = 0; c < d; ++c) {
                    result(r, c) = (x(r, c) / rms) * static_cast<T>(g[c]);
                }
            }
        };

        const std::size_t worker_count = std::min(pool.size(), seq_len);
        const std::size_t rows_per_worker = (seq_len + worker_count - 1) / worker_count;
        pool.run(worker_count, [&](std::size_t t) {
            const std::size_t r_start = t * rows_per_worker;
            worker(r_start, std::min(r_start + rows_per_worker, seq_len));
        });

        return result;
    }

private:
    const MMapWeight* gamma_;
    T eps_;
};

// Full LLaMA transformer block backed by mmap weights:
//   h   = x + GQA_MMap(RMSNorm_MMap(x))
//   out = h + SwiGLU_MMap(RMSNorm_MMap(h))
// Borrows all weights from an MMapLlamaBlockWeights and its thread pool —
// the caller keeps both alive.
template <typename T>
class LlamaBlock_MMap {
public:
    LlamaBlock_MMap(std::size_t n_heads, std::size_t n_kv_heads, std::size_t head_dim,
                    const MMapLlamaBlockWeights& w, ThreadPool& pool,
                    float rope_theta = 5000000.0f, bool use_rope = true)
        : attn_norm_{&w.input_layernorm},
          attn_{n_heads, n_kv_heads, head_dim,
                &w.q_proj, &w.k_proj, &w.v_proj, &w.o_proj, rope_theta, use_rope},
          ffn_norm_{&w.post_attention_layernorm},
          mlp_{&w.gate_proj, &w.up_proj, &w.down_proj},
          pool_{pool} {}

    // x: (seq_len, d_model) -> (seq_len, d_model)
    // cache/layer: see GQA_MMap::forward. The caller advances the cache once all layers ran.
    Mat<T> forward(const Mat<T>& x, KVCache<T>* cache = nullptr, std::size_t layer = 0) {
        Mat<T> h = x.add(attn_.forward(attn_norm_.forward(x, pool_), pool_, cache, layer));
        return h.add(mlp_.forward(ffn_norm_.forward(h, pool_), pool_));
    }

private:
    RMSNorm_MMap<T> attn_norm_;
    GQA_MMap<T> attn_;
    RMSNorm_MMap<T> ffn_norm_;
    SwiGLU_MMap<T> mlp_;
    ThreadPool& pool_;
};
