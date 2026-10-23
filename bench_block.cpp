#include "llama_transformer.h"

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <random>
#include <vector>

static Mat<float> random_mat(std::size_t rows, std::size_t cols, std::mt19937& rng) {
    std::normal_distribution<float> dist(0.0f, 0.02f);
    std::vector<float> data(rows * cols);
    for (auto& v : data) v = dist(rng);
    return Mat<float>(rows, cols, std::move(data));
}

static Mat<float> ones_row(std::size_t cols) {
    return Mat<float>(1, cols, std::vector<float>(cols, 1.0f));
}

int main() {
    constexpr std::size_t d_model   = 512;
    constexpr std::size_t n_heads   = 8;
    constexpr std::size_t n_kv_heads = 4;
    constexpr std::size_t head_dim  = d_model / n_heads; // 64
    constexpr std::size_t d_ff      = 1376; // ≈ 8/3 * d_model, rounded to even
    constexpr std::size_t seq_len   = 500;

    std::mt19937 rng(42);

    std::printf("Building LlamaBlock: d_model=%zu, n_heads=%zu, n_kv_heads=%zu, "
                "head_dim=%zu, d_ff=%zu, seq_len=%zu\n",
                d_model, n_heads, n_kv_heads, head_dim, d_ff, seq_len);

    // Weights
    auto W_q = random_mat(d_model, n_heads * head_dim, rng);
    auto W_k = random_mat(d_model, n_kv_heads * head_dim, rng);
    auto W_v = random_mat(d_model, n_kv_heads * head_dim, rng);
    auto W_o = random_mat(n_heads * head_dim, d_model, rng);

    auto W_gate = random_mat(d_model, d_ff, rng);
    auto W_up   = random_mat(d_model, d_ff, rng);
    auto W_down  = random_mat(d_ff, d_model, rng);

    RMSNorm<float> attn_norm(ones_row(d_model));
    GQA<float> attn(n_heads, n_kv_heads, head_dim,
                    std::move(W_q), std::move(W_k),
                    std::move(W_v), std::move(W_o));
    RMSNorm<float> ffn_norm(ones_row(d_model));
    SwiGLU<float> mlp(std::move(W_gate), std::move(W_up), std::move(W_down));

    LlamaBlock<float> block(std::move(attn_norm), std::move(attn),
                            std::move(ffn_norm), std::move(mlp));

    // Input: random "logits" as if coming from an embedding layer
    auto x = random_mat(seq_len, d_model, rng);

    std::printf("Running forward pass...\n");

    auto t0 = std::chrono::high_resolution_clock::now();
    auto out = block.forward(x);
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("Done. Output shape: (%zu, %zu)\n", out.rows(), out.cols());
    std::printf("Wall time: %.1f ms\n", ms);

    // Print a few output values as a sanity check
    std::printf("out[0][0..3] = %.6f  %.6f  %.6f  %.6f\n",
                out(0, 0), out(0, 1), out(0, 2), out(0, 3));
    std::printf("out[%zu][0..3] = %.6f  %.6f  %.6f  %.6f\n",
                seq_len - 1,
                out(seq_len - 1, 0), out(seq_len - 1, 1),
                out(seq_len - 1, 2), out(seq_len - 1, 3));

    return 0;
}
