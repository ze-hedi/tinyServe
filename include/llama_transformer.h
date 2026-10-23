#pragma once

#include "mat.h"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

// Apply Rotary Position Embeddings (RoPE) in-place to x: (seq_len, head_dim).
// Each row i corresponds to token at position (pos_offset + i).
// Pairs of dimensions (2j, 2j+1) are rotated by angle:
//   theta_j = (pos_offset + i) / 10000^(2j / head_dim)
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
            T theta = static_cast<T>(pos) /
                      std::pow(T{10000}, static_cast<T>(2 * j) / static_cast<T>(head_dim));
            T cos_t = std::cos(theta);
            T sin_t = std::sin(theta);
            T x0 = x(i, 2 * j);
            T x1 = x(i, 2 * j + 1);
            x(i, 2 * j)     = x0 * cos_t - x1 * sin_t;
            x(i, 2 * j + 1) = x0 * sin_t + x1 * cos_t;
        }
    }
}

// Grouped Query Attention (GQA) — LLaMA style.
// n_heads query heads, n_kv_heads key/value heads (n_heads must be divisible by n_kv_heads).
// Causal (autoregressive) masking applied.
template <typename T>
class GQA {
public:
    // W_q: (d_model, n_heads * head_dim)
    // W_k: (d_model, n_kv_heads * head_dim)
    // W_v: (d_model, n_kv_heads * head_dim)
    // W_o: (n_heads * head_dim, d_model)
    GQA(std::size_t n_heads, std::size_t n_kv_heads, std::size_t head_dim,
         Mat<T> W_q, Mat<T> W_k, Mat<T> W_v, Mat<T> W_o)
        : n_heads_{n_heads}, n_kv_heads_{n_kv_heads}, head_dim_{head_dim},
          W_q_{std::move(W_q)}, W_k_{std::move(W_k)},
          W_v_{std::move(W_v)}, W_o_{std::move(W_o)} {
        if (n_heads_ % n_kv_heads_ != 0) {
            throw std::invalid_argument("n_heads must be divisible by n_kv_heads");
        }
        std::size_t d = W_q_.rows();
        if (W_q_.cols() != n_heads_ * head_dim_) {
            throw std::invalid_argument("W_q cols must equal n_heads * head_dim");
        }
        if (W_k_.rows() != d || W_k_.cols() != n_kv_heads_ * head_dim_) {
            throw std::invalid_argument("W_k dimensions mismatch");
        }
        if (W_v_.rows() != d || W_v_.cols() != n_kv_heads_ * head_dim_) {
            throw std::invalid_argument("W_v dimensions mismatch");
        }
        if (W_o_.rows() != n_heads_ * head_dim_ || W_o_.cols() != d) {
            throw std::invalid_argument("W_o dimensions mismatch");
        }
    }

    std::size_t n_heads() const { return n_heads_; }
    std::size_t n_kv_heads() const { return n_kv_heads_; }
    std::size_t head_dim() const { return head_dim_; }
    std::size_t d_model() const { return W_q_.rows(); }

    // x: (seq_len, d_model) -> output: (seq_len, d_model)
    Mat<T> forward(const Mat<T>& x) const {
        if (x.cols() != d_model()) {
            throw std::invalid_argument("input cols must equal d_model");
        }

        std::size_t seq_len = x.rows();
        std::size_t n_groups = n_heads_ / n_kv_heads_;

        // Project
        Mat<T> Q = x.matmul(W_q_); // (seq_len, n_heads * head_dim)
        Mat<T> K = x.matmul(W_k_); // (seq_len, n_kv_heads * head_dim)
        Mat<T> V = x.matmul(W_v_); // (seq_len, n_kv_heads * head_dim)

        T scale = T{1} / std::sqrt(static_cast<T>(head_dim_));

        // Output buffer: (seq_len, n_heads * head_dim)
        Mat<T> concat(seq_len, n_heads_ * head_dim_);

        // Per-head attention
        for (std::size_t h = 0; h < n_heads_; ++h) {
            std::size_t kv_h = h / n_groups;

            // Q_h, K_h, V_h: (seq_len, head_dim)
            Mat<T> Q_h = Q.slice_cols(h * head_dim_, head_dim_);
            Mat<T> K_h = K.slice_cols(kv_h * head_dim_, head_dim_);
            Mat<T> V_h = V.slice_cols(kv_h * head_dim_, head_dim_);

            // Apply RoPE to Q and K before attention
            apply_rope(Q_h);
            apply_rope(K_h);

            // scores = Q_h @ K_h^T / sqrt(head_dim) -> (seq_len, seq_len)
            Mat<T> scores = Q_h.matmul(K_h.transpose());
            scores.scale(scale);

            // Causal mask: zero out future positions
            scores.masked_fill(-std::numeric_limits<T>::infinity(), true);

            // Softmax each row
            scores.softmax_rows();

            // attn_out = scores @ V_h -> (seq_len, head_dim)
            Mat<T> attn_out = scores.matmul(V_h);

            // Write into concat
            concat.set_cols(h * head_dim_, attn_out);
        }

        // Output projection: (seq_len, n_heads * head_dim) @ W_o -> (seq_len, d_model)
        return concat.matmul(W_o_);
    }

private:
    std::size_t n_heads_;
    std::size_t n_kv_heads_;
    std::size_t head_dim_;
    Mat<T> W_q_;
    Mat<T> W_k_;
    Mat<T> W_v_;
    Mat<T> W_o_;
};

// RMSNorm: x_norm = (x / rms(x)) * gamma, where rms(x) = sqrt(mean(x^2) + eps)
// gamma: learnable per-dimension scale, size d_model
template <typename T>
class RMSNorm {
public:
    // gamma: (1, d_model) row vector of scale weights
    RMSNorm(Mat<T> gamma, T eps = T{1e-6})
        : gamma_{std::move(gamma)}, eps_{eps} {
        if (gamma_.rows() != 1) {
            throw std::invalid_argument("RMSNorm: gamma must be a (1, d_model) row vector");
        }
    }

    std::size_t d_model() const { return gamma_.cols(); }

    // x: (seq_len, d_model) -> (seq_len, d_model)
    Mat<T> forward(const Mat<T>& x) const {
        if (x.cols() != d_model()) {
            throw std::invalid_argument("RMSNorm: input cols must equal d_model");
        }
        std::size_t seq_len = x.rows();
        std::size_t d = d_model();
        Mat<T> result(seq_len, d);
        for (std::size_t r = 0; r < seq_len; ++r) {
            T sum_sq{0};
            for (std::size_t c = 0; c < d; ++c) {
                sum_sq += x(r, c) * x(r, c);
            }
            T rms = std::sqrt(sum_sq / static_cast<T>(d) + eps_);
            for (std::size_t c = 0; c < d; ++c) {
                result(r, c) = (x(r, c) / rms) * gamma_(0, c);
            }
        }
        return result;
    }

private:
    Mat<T> gamma_;
    T eps_;
};

// SwiGLU MLP (LLaMA FFN): out = (silu(x @ W_gate) ⊙ (x @ W_up)) @ W_down
// silu(z) = z * sigmoid(z)
template <typename T>
class SwiGLU {
public:
    // W_gate: (d_model, d_ff)
    // W_up:   (d_model, d_ff)
    // W_down: (d_ff, d_model)
    SwiGLU(Mat<T> W_gate, Mat<T> W_up, Mat<T> W_down)
        : W_gate_{std::move(W_gate)}, W_up_{std::move(W_up)}, W_down_{std::move(W_down)} {
        std::size_t d = W_gate_.rows();
        std::size_t ff = W_gate_.cols();
        if (W_up_.rows() != d || W_up_.cols() != ff) {
            throw std::invalid_argument("SwiGLU: W_up dimensions must match W_gate");
        }
        if (W_down_.rows() != ff || W_down_.cols() != d) {
            throw std::invalid_argument("SwiGLU: W_down must be (d_ff, d_model)");
        }
    }

    std::size_t d_model() const { return W_gate_.rows(); }
    std::size_t d_ff() const { return W_gate_.cols(); }

    // x: (seq_len, d_model) -> (seq_len, d_model)
    Mat<T> forward(const Mat<T>& x) const {
        if (x.cols() != d_model()) {
            throw std::invalid_argument("SwiGLU: input cols must equal d_model");
        }
        Mat<T> gate = x.matmul(W_gate_);  // (seq_len, d_ff)
        Mat<T> up = x.matmul(W_up_);      // (seq_len, d_ff)

        // silu(gate) ⊙ up
        Mat<T> activated = gate.apply([](T z) {
            return z / (T{1} + std::exp(-z));  // silu(z) = z * sigmoid(z)
        });
        Mat<T> fused = activated.hadamard(up);

        return fused.matmul(W_down_);  // (seq_len, d_model)
    }

private:
    Mat<T> W_gate_;
    Mat<T> W_up_;
    Mat<T> W_down_;
};

// Full LLaMA transformer block:
//   h   = x + GQA(RMSNorm_attn(x))
//   out = h + SwiGLU(RMSNorm_ffn(h))
template <typename T>
class LlamaBlock {
public:
    LlamaBlock(RMSNorm<T> attn_norm, GQA<T> attn,
               RMSNorm<T> ffn_norm, SwiGLU<T> mlp)
        : attn_norm_{std::move(attn_norm)}, attn_{std::move(attn)},
          ffn_norm_{std::move(ffn_norm)}, mlp_{std::move(mlp)} {}

    // x: (seq_len, d_model) -> (seq_len, d_model)
    Mat<T> forward(const Mat<T>& x) const {
        Mat<T> h = x.add(attn_.forward(attn_norm_.forward(x)));
        return h.add(mlp_.forward(ffn_norm_.forward(h)));
    }

private:
    RMSNorm<T> attn_norm_;
    GQA<T> attn_;
    RMSNorm<T> ffn_norm_;
    SwiGLU<T> mlp_;
};
