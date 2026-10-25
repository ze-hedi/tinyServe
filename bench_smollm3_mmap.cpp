#include "llama_mmap_block.h"
#include "llama_transformer.h"
#include "smollm3.h"
#include "thread_pool.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    std::string model_root = "smollm3";
    std::size_t seq_len = 200;
    std::size_t layers = 0;  // 0 -> SmolLM3 default (36)
    std::size_t threads = mat_detail::hardware_thread_count();
    std::size_t warmup = 0;
    std::size_t iterations = 1;
    MMapLoadOptions mmap_options;
    bool check_reference = false;
};

struct Faults {
    long minor = 0;
    long major = 0;
};

[[noreturn]] void usage(const char* program, int status = 2) {
    std::fprintf(stderr,
                 "Usage: %s [model_root] [--seq N] [--layers N] [--threads N] "
                 "[--warmup N] [--iters N] [--prefault] [--hugepages] [--check]\n",
                 program);
    std::exit(status);
}

std::size_t parse_size(const char* value, const char* option, bool allow_zero = false) {
    std::string text(value);
    if (text.empty() ||
        !std::all_of(text.begin(), text.end(), [](unsigned char c) { return std::isdigit(c) != 0; })) {
        throw std::invalid_argument(std::string("invalid value for ") + option);
    }
    std::size_t parsed = 0;
    const auto result = std::stoull(text, &parsed);
    if (parsed != text.size() || result > std::numeric_limits<std::size_t>::max() ||
        (!allow_zero && result == 0)) {
        throw std::invalid_argument(std::string("invalid value for ") + option);
    }
    return result;
}

Options parse_options(int argc, char* argv[]) {
    Options options;
    bool has_model_root = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--help") {
            usage(argv[0], 0);
        }
        if (!arg.starts_with("--")) {
            if (has_model_root) usage(argv[0]);
            options.model_root = arg;
            has_model_root = true;
            continue;
        }
        if (arg == "--check") {
            options.check_reference = true;
            continue;
        }
        if (arg == "--prefault") {
            options.mmap_options.prefault = true;
            continue;
        }
        if (arg == "--hugepages") {
            options.mmap_options.huge_pages = true;
            continue;
        }
        if (i + 1 >= argc) usage(argv[0]);
        const char* value = argv[++i];
        if (arg == "--seq") {
            options.seq_len = parse_size(value, "--seq");
        } else if (arg == "--layers") {
            options.layers = parse_size(value, "--layers", true);
        } else if (arg == "--threads") {
            options.threads = parse_size(value, "--threads");
        } else if (arg == "--warmup") {
            options.warmup = parse_size(value, "--warmup", true);
        } else if (arg == "--iters") {
            options.iterations = parse_size(value, "--iters");
        } else {
            usage(argv[0]);
        }
    }
    return options;
}

Mat<float> random_mat(std::size_t rows, std::size_t cols, std::mt19937& rng) {
    std::normal_distribution<float> dist(0.0f, 0.02f);
    std::vector<float> data(rows * cols);
    for (auto& v : data) v = dist(rng);
    return Mat<float>(rows, cols, std::move(data));
}

Mat<float> copy_weight(const MMapWeight& weight) {
    const float* source = weight.data();
    return Mat<float>(weight.rows(), weight.cols(),
                      std::vector<float>(source, source + weight.rows() * weight.cols()));
}

Faults faults() {
    rusage usage_info{};
    if (::getrusage(RUSAGE_SELF, &usage_info) != 0) {
        throw std::runtime_error("getrusage failed");
    }
    return {usage_info.ru_minflt, usage_info.ru_majflt};
}

double checksum(const Mat<float>& matrix) {
    double sum = 0.0;
    for (std::size_t r = 0; r < matrix.rows(); ++r) {
        for (std::size_t c = 0; c < matrix.cols(); ++c) {
            sum += matrix(r, c);
        }
    }
    return sum;
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        const auto options = parse_options(argc, argv);
        SmolLM3<float>::Config config;
        if (options.layers != 0) {
            config.n_layers = options.layers;
        }

        const auto load_faults_before = faults();
        const auto load_start = Clock::now();
        SmolLM3<float> model(options.model_root, config, options.mmap_options, options.threads);
        const auto load_end = Clock::now();
        const auto load_faults_after = faults();
        const double load_ms = std::chrono::duration<double, std::milli>(load_end - load_start).count();

        std::mt19937 rng(42);
        const auto x = random_mat(options.seq_len, config.d_model, rng);

        std::printf(
            "config: layers=%zu d_model=%zu heads=%zu kv_heads=%zu head_dim=%zu seq=%zu "
            "threads=%zu warmup=%zu iters=%zu prefault=%d hugepage_advice=%d "
            "load_ms=%.3f load_minor_faults=%ld load_major_faults=%ld\n",
            config.n_layers, config.d_model, config.n_heads, config.n_kv_heads, config.head_dim,
            options.seq_len, options.threads, options.warmup, options.iterations,
            options.mmap_options.prefault, options.mmap_options.huge_pages, load_ms,
            load_faults_after.minor - load_faults_before.minor,
            load_faults_after.major - load_faults_before.major);

        Mat<float> out(0, 0);
        for (std::size_t i = 0; i < options.warmup; ++i) {
            const auto start = Clock::now();
            out = model.forward(x);
            const auto end = Clock::now();
            const double ms = std::chrono::duration<double, std::milli>(end - start).count();
            std::printf("warmup[%zu]: %.3f ms checksum=%.9g\n", i, ms, checksum(out));
        }

        std::vector<double> samples;
        samples.reserve(options.iterations);
        for (std::size_t i = 0; i < options.iterations; ++i) {
            const auto faults_before = faults();
            const auto start = Clock::now();
            out = model.forward(x);
            const auto end = Clock::now();
            const auto faults_after = faults();
            const double ms = std::chrono::duration<double, std::milli>(end - start).count();
            samples.push_back(ms);
            std::printf(
                "iteration[%zu]: %.3f ms checksum=%.9g minor_faults=%ld major_faults=%ld\n",
                i, ms, checksum(out), faults_after.minor - faults_before.minor,
                faults_after.major - faults_before.major);
        }

        std::vector<double> sorted = samples;
        std::sort(sorted.begin(), sorted.end());
        const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
        const double median = sorted.size() % 2 == 0
            ? (sorted[sorted.size() / 2 - 1] + sorted[sorted.size() / 2]) / 2.0
            : sorted[sorted.size() / 2];
        std::printf("summary: median=%.3f ms mean=%.3f ms min=%.3f ms max=%.3f ms\n",
                    median, mean, sorted.front(), sorted.back());
        std::printf("output: shape=(%zu,%zu) first=%.6f last=%.6f\n",
                    out.rows(), out.cols(), out(0, 0), out(out.rows() - 1, 0));

        if (options.check_reference) {
            ThreadPool pool(options.threads);
            std::vector<std::unique_ptr<LlamaBlock<float>>> reference_blocks;
            reference_blocks.reserve(config.n_layers);
            const std::filesystem::path root(options.model_root);
            for (std::size_t layer = 0; layer < config.n_layers; ++layer) {
                const auto block_dir = root / ("bloc_" + std::to_string(layer));
                auto weights = load_mapped_llama_block(block_dir, options.mmap_options);
                reference_blocks.push_back(std::make_unique<LlamaBlock<float>>(
                    RMSNorm<float>(copy_weight(weights.input_layernorm)),
                    GQA<float>(config.n_heads, config.n_kv_heads, config.head_dim,
                               copy_weight(weights.q_proj), copy_weight(weights.k_proj),
                               copy_weight(weights.v_proj), copy_weight(weights.o_proj)),
                    RMSNorm<float>(copy_weight(weights.post_attention_layernorm)),
                    SwiGLU<float>(copy_weight(weights.gate_proj), copy_weight(weights.up_proj),
                                  copy_weight(weights.down_proj)),
                    pool));
            }
            Mat<float> expected = reference_blocks[0]->forward(x);
            for (std::size_t layer = 1; layer < reference_blocks.size(); ++layer) {
                expected = reference_blocks[layer]->forward(expected);
            }
            double max_abs_error = 0.0;
            double max_relative_error = 0.0;
            for (std::size_t r = 0; r < out.rows(); ++r) {
                for (std::size_t c = 0; c < out.cols(); ++c) {
                    const double error = std::abs(static_cast<double>(out(r, c)) - expected(r, c));
                    max_abs_error = std::max(max_abs_error, error);
                    const double relative = error / std::max(1e-12, std::abs(static_cast<double>(expected(r, c))));
                    max_relative_error = std::max(max_relative_error, relative);
                    if (!std::isfinite(out(r, c)) || !std::isfinite(expected(r, c)) ||
                        error > 1e-4 + 1e-4 * std::abs(expected(r, c))) {
                        throw std::runtime_error("mmap output differs from the non-mmap reference");
                    }
                }
            }
            std::printf("reference: max_abs_error=%.9g max_relative_error=%.9g status=ok\n",
                        max_abs_error, max_relative_error);
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "bench_smollm3_mmap: %s\n", error.what());
        return 1;
    }
}
