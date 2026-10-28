#include "embeddings_mmap.h"
#include "gpu_transformer.h"
#include "kv_cache.h"
#include "llama_mmap_block.h"
#include "mat.h"
#include "mmap_block_loader.h"
#include "thread_pool.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

// Checks the GPU decode step against the CPU blocks on real SmolLM3 weights.
//   1. The CPU prefills a prompt (embeddings -> first N blocks) and fills its KV cache.
//   2. That cache is copied to the GPU, so both sides start from identical state.
//   3. For each step, the same token embedding goes through the CPU blocks and the GPU
//      transformer. Compared: the output hidden state and the K/V row each layer appended.
// Error is measured as max|gpu - cpu| / max|cpu|. Exit code 1 if any step exceeds --tol.

using Clock = std::chrono::steady_clock;

static double ms_since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

static double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// max|a - b| / max|b|
static double relative_error(const float* a, const float* b, std::size_t n) {
    double max_diff = 0.0;
    double max_ref = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        max_diff = std::max(max_diff, std::fabs(static_cast<double>(a[i]) - b[i]));
        max_ref = std::max(max_ref, std::fabs(static_cast<double>(b[i])));
    }
    return max_ref == 0.0 ? max_diff : max_diff / max_ref;
}

int main(int argc, char* argv[]) {
    try {
        std::filesystem::path model_root = "smollm3";
        std::size_t n_layers = 4;  // includes layer 3, the first NoPE layer
        std::size_t prompt_len = 16;
        std::size_t steps = 8;
        std::size_t threads = mat_detail::default_thread_count();
        double tol = 1e-3;

        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            const bool has_value = i + 1 < argc;
            if (arg == "--layers" && has_value) n_layers = std::stoull(argv[++i]);
            else if (arg == "--prompt" && has_value) prompt_len = std::stoull(argv[++i]);
            else if (arg == "--steps" && has_value) steps = std::stoull(argv[++i]);
            else if (arg == "--threads" && has_value) threads = std::stoull(argv[++i]);
            else if (arg == "--tol" && has_value) tol = std::stod(argv[++i]);
            else if (!arg.starts_with("--")) model_root = arg;
            else {
                std::fprintf(stderr,
                             "usage: gpu_decode_check [model_dir] [--layers N] [--prompt P] "
                             "[--steps S] [--threads T] [--tol X]\n");
                return 2;
            }
        }
        if (prompt_len == 0 || steps == 0 || n_layers == 0) {
            std::fprintf(stderr, "--layers, --prompt and --steps must be positive\n");
            return 2;
        }

        // SmolLM3-3B geometry.
        const std::size_t n_heads = 16, n_kv_heads = 4, head_dim = 128;
        const std::size_t d_model = n_heads * head_dim;
        const std::size_t kv_dim = n_kv_heads * head_dim;
        const float rope_theta = 5000000.0f;
        const std::size_t capacity = prompt_len + steps;

        std::printf("Loading %zu blocks from '%s'...\n", n_layers, model_root.string().c_str());
        MMapWeight embed_weights =
            load_mapped_single_tensor(model_root, "embed_tokens.json", "embed_tokens");
        Embedding_MMap<float> embedding(&embed_weights);

        std::vector<MMapLlamaBlockWeights> weights(n_layers);  // sized once: blocks borrow it
        for (std::size_t i = 0; i < n_layers; ++i) {
            weights[i] = load_mapped_llama_block(model_root / ("bloc_" + std::to_string(i)));
        }

        // CPU reference.
        ThreadPool pool(threads);
        std::vector<std::unique_ptr<LlamaBlock_MMap<float>>> cpu_blocks;
        for (std::size_t i = 0; i < n_layers; ++i) {
            cpu_blocks.push_back(std::make_unique<LlamaBlock_MMap<float>>(
                n_heads, n_kv_heads, head_dim, weights[i], pool, rope_theta, i % 4 != 3));
        }
        KVCache<float> cpu_cache(n_layers, kv_dim, capacity);

        // GPU under test.
        gpu::GpuModelConfig config;
        std::vector<gpu::GpuLayerSource> sources;
        for (std::size_t i = 0; i < n_layers; ++i) sources.push_back({i, &weights[i]});
        auto upload_start = Clock::now();
        gpu::GpuTransformer gpu_model(config, sources, capacity);
        std::printf("Uploaded weights to the GPU in %.0f ms\n", ms_since(upload_start));

        // Deterministic token ids: prompt tokens followed by one token per decode step.
        const std::size_t vocab = embedding.vocab_size();
        std::vector<std::size_t> ids(prompt_len + steps);
        for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = (1000 + 7919 * i) % vocab;

        // 1-2. CPU prefill, then hand the cache to the GPU.
        Mat<float> hidden = embedding.forward(
            std::vector<std::size_t>(ids.begin(), ids.begin() + prompt_len));
        for (std::size_t i = 0; i < n_layers; ++i) {
            hidden = cpu_blocks[i]->forward(hidden, &cpu_cache, i);
        }
        cpu_cache.advance(prompt_len);
        for (std::size_t i = 0; i < n_layers; ++i) {
            gpu_model.cache().upload_rows(i, 0, prompt_len, cpu_cache.k(i), cpu_cache.v(i));
        }
        std::printf("Prefilled %zu tokens on the CPU, KV cache copied to the GPU\n\n",
                    prompt_len);

        // 3. Decode steps.
        std::printf("%5s  %12s  %12s  %9s  %9s\n", "step", "hidden err", "kv err", "cpu ms",
                    "gpu ms");
        std::vector<double> cpu_times;
        std::vector<double> gpu_times;
        double worst = 0.0;
        std::vector<float> input(d_model), gpu_out(d_model), cpu_out(d_model);
        std::vector<float> gpu_k(kv_dim), gpu_v(kv_dim);

        for (std::size_t s = 0; s < steps; ++s) {
            const std::size_t pos = prompt_len + s;
            Mat<float> x = embedding.forward({ids[pos]});
            for (std::size_t c = 0; c < d_model; ++c) input[c] = x(0, c);

            auto t0 = Clock::now();
            Mat<float> h = std::move(x);
            for (std::size_t i = 0; i < n_layers; ++i) h = cpu_blocks[i]->forward(h, &cpu_cache, i);
            cpu_cache.advance(1);
            const double cpu_ms = ms_since(t0);
            for (std::size_t c = 0; c < d_model; ++c) cpu_out[c] = h(0, c);

            t0 = Clock::now();
            gpu_model.decode(input.data(), gpu_out.data(), pos);
            const double gpu_ms = ms_since(t0);

            const double hidden_err = relative_error(gpu_out.data(), cpu_out.data(), d_model);
            double kv_err = 0.0;
            for (std::size_t i = 0; i < n_layers; ++i) {
                gpu_model.cache().download_rows(i, pos, 1, gpu_k.data(), gpu_v.data());
                kv_err = std::max(kv_err, relative_error(gpu_k.data(),
                                                         cpu_cache.k(i) + pos * kv_dim, kv_dim));
                kv_err = std::max(kv_err, relative_error(gpu_v.data(),
                                                         cpu_cache.v(i) + pos * kv_dim, kv_dim));
            }

            std::printf("%5zu  %12.3e  %12.3e  %9.2f  %9.2f%s\n", s, hidden_err, kv_err, cpu_ms,
                        gpu_ms, s == 0 ? "  (first GPU call: warm-up)" : "");
            worst = std::max({worst, hidden_err, kv_err});
            cpu_times.push_back(cpu_ms);
            if (s > 0) gpu_times.push_back(gpu_ms);
        }

        std::printf("\nmedian per step over %zu layers: CPU %.2f ms, GPU %.2f ms (incl. 2 x %zu "
                    "bytes of host<->device copies)\n",
                    n_layers, median(cpu_times), median(gpu_times), d_model * sizeof(float));
        std::printf("worst relative error %.3e (tolerance %.1e): %s\n", worst, tol,
                    worst <= tol ? "PASS" : "FAIL");
        return worst <= tol ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "gpu_decode_check: %s\n", e.what());
        return 1;
    }
}
