#include "llama_mmap_block.h"

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

int main(int argc, char* argv[]) {
    const char* block_dir = "bloc_0";
    if (argc > 1) block_dir = argv[1];

    constexpr std::size_t d_model    = 2048;
    constexpr std::size_t n_heads    = 16;
    constexpr std::size_t n_kv_heads = 4;
    constexpr std::size_t head_dim   = d_model / n_heads; // 128
    constexpr std::size_t seq_len    = 200;

    std::printf("Loading mmap weights from '%s'...\n", block_dir);
    auto weights = load_mapped_llama_block(block_dir);

    std::printf("Building LlamaBlock_MMap: d_model=%zu, n_heads=%zu, n_kv_heads=%zu, "
                "head_dim=%zu, seq_len=%zu\n",
                d_model, n_heads, n_kv_heads, head_dim, seq_len);

    LlamaBlock_MMap<float> block(n_heads, n_kv_heads, head_dim, weights);

    // Input: random "logits" as if coming from an embedding layer
    std::mt19937 rng(42);
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
