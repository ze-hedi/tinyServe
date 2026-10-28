#include "embeddings_mmap.h"
#include "gpu_head.h"
#include "llama_mmap_block.h"
#include "mat.h"
#include "mmap_block_loader.h"
#include "thread_pool.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <vector>

// Checks the GPU head (final RMSNorm + tied lm_head) against the CPU path used by SmolLM3.
// Test vectors are real hidden states: a short prompt goes through the first --layers CPU
// blocks and every row is fed to both heads. Compared per vector: the full logits row
// (max|gpu - cpu| / max|cpu|), the arg-max token and the top-5 token set. The CPU and GPU
// times of the head itself are reported too. Exit code 1 on any mismatch.

using Clock = std::chrono::steady_clock;

static double ms_since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

static double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

static std::vector<std::size_t> top_k(const float* x, std::size_t n, std::size_t k) {
    std::vector<std::size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](std::size_t a, std::size_t b) { return x[a] > x[b]; });
    idx.resize(k);
    return idx;
}

int main(int argc, char* argv[]) {
    try {
        std::filesystem::path model_root = "smollm3";
        std::size_t n_layers = 4;
        std::size_t vectors = 8;
        std::size_t threads = mat_detail::default_thread_count();
        double tol = 1e-3;

        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            const bool has_value = i + 1 < argc;
            if (arg == "--layers" && has_value) n_layers = std::stoull(argv[++i]);
            else if (arg == "--vectors" && has_value) vectors = std::stoull(argv[++i]);
            else if (arg == "--threads" && has_value) threads = std::stoull(argv[++i]);
            else if (arg == "--tol" && has_value) tol = std::stod(argv[++i]);
            else if (!arg.starts_with("--")) model_root = arg;
            else {
                std::fprintf(stderr, "usage: gpu_head_check [model_dir] [--layers N] "
                                     "[--vectors V] [--threads T] [--tol X]\n");
                return 2;
            }
        }
        if (vectors < 2 || n_layers == 0) {
            std::fprintf(stderr, "--vectors must be >= 2 and --layers positive\n");
            return 2;
        }

        const std::size_t n_heads = 16, n_kv_heads = 4, head_dim = 128;
        const float rope_theta = 5000000.0f;

        std::printf("Loading embedding, final norm and %zu blocks from '%s'...\n", n_layers,
                    model_root.string().c_str());
        MMapWeight embed_weights =
            load_mapped_single_tensor(model_root, "embed_tokens.json", "embed_tokens");
        MMapWeight final_norm_weights =
            load_mapped_single_tensor(model_root, "final_norm.json", "final_norm");
        Embedding_MMap<float> embedding(&embed_weights);
        const std::size_t d = embed_weights.in_features();
        const std::size_t vocab = embed_weights.out_features();

        std::vector<MMapLlamaBlockWeights> weights(n_layers);
        for (std::size_t i = 0; i < n_layers; ++i) {
            weights[i] = load_mapped_llama_block(model_root / ("bloc_" + std::to_string(i)));
        }

        // CPU side: real hidden states, and the reference head exactly as SmolLM3 computes it.
        ThreadPool pool(threads);
        Mat<float> hidden = embedding.forward([&] {
            std::vector<std::size_t> ids(vectors);
            for (std::size_t i = 0; i < vectors; ++i) ids[i] = (1000 + 7919 * i) % vocab;
            return ids;
        }());
        for (std::size_t i = 0; i < n_layers; ++i) {
            LlamaBlock_MMap<float> block(n_heads, n_kv_heads, head_dim, weights[i], pool,
                                         rope_theta, i % 4 != 3);
            hidden = block.forward(hidden);
        }
        RMSNorm_MMap<float> final_norm(&final_norm_weights);

        auto upload_start = Clock::now();
        gpu::GpuHead head(final_norm_weights, embed_weights);
        std::printf("Uploaded lm_head (%.0f MiB) in %.0f ms\n\n",
                    d * vocab * sizeof(float) / 1048576.0, ms_since(upload_start));

        std::printf("%4s  %12s  %7s  %9s  %9s  %9s\n", "vec", "logits err", "top-1", "top-5",
                    "cpu ms", "gpu ms");
        std::vector<double> cpu_times, gpu_times;
        std::vector<float> row(d), gpu_logits(vocab);
        bool all_ok = true;
        double worst = 0.0;

        for (std::size_t r = 0; r < vectors; ++r) {
            for (std::size_t c = 0; c < d; ++c) row[c] = hidden(r, c);

            // CPU path: norm the row, then project (same order as SmolLM3::lm_head).
            auto t0 = Clock::now();
            Mat<float> normed = final_norm.forward(Mat<float>(1, d, row), pool);
            Mat<float> cpu_logits = normed.matmul(embed_weights, pool);
            const double cpu_ms = ms_since(t0);

            t0 = Clock::now();
            head.logits(row.data(), gpu_logits.data());
            const double gpu_ms = ms_since(t0);

            double max_diff = 0.0, max_ref = 0.0;
            for (std::size_t v = 0; v < vocab; ++v) {
                max_diff = std::max(max_diff, std::fabs(static_cast<double>(gpu_logits[v]) -
                                                        cpu_logits(0, v)));
                max_ref = std::max(max_ref, std::fabs(static_cast<double>(cpu_logits(0, v))));
            }
            const double err = max_ref == 0.0 ? max_diff : max_diff / max_ref;

            std::vector<float> cpu_row(vocab);
            for (std::size_t v = 0; v < vocab; ++v) cpu_row[v] = cpu_logits(0, v);
            const auto cpu_top = top_k(cpu_row.data(), vocab, 5);
            const auto gpu_top = top_k(gpu_logits.data(), vocab, 5);
            const bool top1 = cpu_top[0] == gpu_top[0];
            const bool top5 = std::set<std::size_t>(cpu_top.begin(), cpu_top.end()) ==
                              std::set<std::size_t>(gpu_top.begin(), gpu_top.end());

            std::printf("%4zu  %12.3e  %7s  %9s  %9.2f  %9.2f%s\n", r, err, top1 ? "same" : "DIFF",
                        top5 ? "same" : "DIFF", cpu_ms, gpu_ms,
                        r == 0 ? "  (first GPU call: warm-up)" : "");
            worst = std::max(worst, err);
            all_ok = all_ok && top1 && top5 && err <= tol;
            if (r > 0) {
                cpu_times.push_back(cpu_ms);
                gpu_times.push_back(gpu_ms);
            }
        }

        std::printf("\nhead median: CPU %.2f ms, GPU %.2f ms (GPU includes the %.0f KiB logits "
                    "copy back)\n",
                    median(cpu_times), median(gpu_times), vocab * sizeof(float) / 1024.0);
        std::printf("worst relative error %.3e (tolerance %.1e): %s\n", worst, tol,
                    all_ok ? "PASS" : "FAIL");
        return all_ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "gpu_head_check: %s\n", e.what());
        return 1;
    }
}
