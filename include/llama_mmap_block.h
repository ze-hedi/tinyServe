#pragma once

#include "mat.h"
#include "mmap_block_loader.h"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

// Apply Rotary Position Embeddings (RoPE) in-place to x: (seq_len, head_dim).
// (same as llama_transformer.h — duplicated to keep this header self-contained)
namespace gqa_mmap_detail {
template <typename T>
void apply_rope(Mat<T>& x, std::size_t pos_offset = 0) {
    std::size_t seq_len = x.rows();
    std::size_t head_dim = x.cols();
    if (head_dim % 2 != 0) {
        throw std::invalid_argument("apply_rope: head_dim must be even");
    }
    for (std::size_t i = 0; i < seq_len; ++i) {
        std::size_t pos = pos_offset + i;
        for (std::size_t j = 0; j < head_dim / 2; ++j) {
            const float theta = static_cast<float>(pos) /
                std::pow(10000.0f,
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
             const MMapWeight* W_v, const MMapWeight* W_o)
        : n_heads_{n_heads}, n_kv_heads_{n_kv_heads}, head_dim_{head_dim},
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
    Mat<T> forward(const Mat<T>& x, std::vector<std::thread>* threads = nullptr) const {
        if (x.cols() != d_model()) {
            throw std::invalid_argument("input cols must equal d_model");
        }

        std::size_t seq_len = x.rows();
        std::size_t n_groups = n_heads_ / n_kv_heads_;

        // Project using mmap weights
        Mat<T> Q = x.matmul(*W_q_, threads); // (seq_len, n_heads * head_dim)
        Mat<T> K = x.matmul(*W_k_, threads); // (seq_len, n_kv_heads * head_dim)
        Mat<T> V = x.matmul(*W_v_, threads); // (seq_len, n_kv_heads * head_dim)

        T scale = T{1} / std::sqrt(static_cast<T>(head_dim_));

        // Output buffer: (seq_len, n_heads * head_dim)
        Mat<T> concat(seq_len, n_heads_ * head_dim_);

        // Per-head attention
        for (std::size_t h = 0; h < n_heads_; ++h) {
            std::size_t kv_h = h / n_groups;

            Mat<T> Q_h = Q.slice_cols(h * head_dim_, head_dim_);
            Mat<T> K_h = K.slice_cols(kv_h * head_dim_, head_dim_);
            Mat<T> V_h = V.slice_cols(kv_h * head_dim_, head_dim_);

            gqa_mmap_detail::apply_rope(Q_h);
            gqa_mmap_detail::apply_rope(K_h);

            // scores = Q_h @ K_h^T / sqrt(head_dim) -> (seq_len, seq_len)
            Mat<T> scores = Q_h.matmul(K_h.transpose(), threads);
            scores.scale(scale);

            scores.masked_fill(-std::numeric_limits<T>::infinity(), true);
            scores.softmax_rows();

            Mat<T> attn_out = scores.matmul(V_h, threads);
            concat.set_cols(h * head_dim_, attn_out);
        }

        // Output projection
        return concat.matmul(*W_o_, threads);
    }

private:
    std::size_t n_heads_;
    std::size_t n_kv_heads_;
    std::size_t head_dim_;
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
    Mat<T> forward(const Mat<T>& x, std::vector<std::thread>* threads = nullptr) const {
        if (x.cols() != d_model()) {
            throw std::invalid_argument("SwiGLU_MMap: input cols must equal d_model");
        }
        Mat<T> gate = x.matmul(*W_gate_, threads);  // (seq_len, d_ff)
        Mat<T> up = x.matmul(*W_up_, threads);      // (seq_len, d_ff)

        // silu(gate) ⊙ up
        Mat<T> activated = gate.apply([](T z) {
            return z / (T{1} + std::exp(-z));
        }, threads);
        Mat<T> fused = activated.hadamard(up);

        return fused.matmul(*W_down_, threads);  // (seq_len, d_model)
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
    Mat<T> forward(const Mat<T>& x, std::vector<std::thread>* threads = nullptr) const {
        if (x.cols() != d_model()) {
            throw std::invalid_argument("RMSNorm_MMap: input cols must equal d_model");
        }
        std::size_t seq_len = x.rows();
        std::size_t d = d_model();
        const float* g = gamma_->data();
        Mat<T> result(seq_len, d);

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

        constexpr std::size_t NUM_THREADS = 30;
        if (threads && seq_len > 1) {
            threads->clear();
            const std::size_t rows_per_thread = (seq_len + NUM_THREADS - 1) / NUM_THREADS;
            for (std::size_t t = 0; t < NUM_THREADS; ++t) {
                const std::size_t r_start = t * rows_per_thread;
                const std::size_t r_end = std::min(r_start + rows_per_thread, seq_len);
                if (r_start >= seq_len) break;
                threads->emplace_back(worker, r_start, r_end);
            }
            for (auto& th : *threads) {
                th.join();
            }
        } else {
            worker(0, seq_len);
        }

        return result;
    }

private:
    const MMapWeight* gamma_;
    T eps_;
};

// Full LLaMA transformer block backed by mmap weights:
//   h   = x + GQA_MMap(RMSNorm_MMap(x))
//   out = h + SwiGLU_MMap(RMSNorm_MMap(h))
// Borrows all weights from an MMapLlamaBlockWeights — caller keeps it alive.
template <typename T>
class LlamaBlock_MMap {
public:
    LlamaBlock_MMap(std::size_t n_heads, std::size_t n_kv_heads, std::size_t head_dim,
                    const MMapLlamaBlockWeights& w)
        : attn_norm_{&w.input_layernorm},
          attn_{n_heads, n_kv_heads, head_dim,
                &w.q_proj, &w.k_proj, &w.v_proj, &w.o_proj},
          ffn_norm_{&w.post_attention_layernorm},
          mlp_{&w.gate_proj, &w.up_proj, &w.down_proj} {
        threads_.reserve(30);
    }

    // x: (seq_len, d_model) -> (seq_len, d_model)
    Mat<T> forward(const Mat<T>& x) {
        Mat<T> h = x.add(attn_.forward(attn_norm_.forward(x, &threads_), &threads_));
        return h.add(mlp_.forward(ffn_norm_.forward(h, &threads_), &threads_));
    }

private:
    RMSNorm_MMap<T> attn_norm_;
    GQA_MMap<T> attn_;
    RMSNorm_MMap<T> ffn_norm_;
    SwiGLU_MMap<T> mlp_;
    std::vector<std::thread> threads_;
};
