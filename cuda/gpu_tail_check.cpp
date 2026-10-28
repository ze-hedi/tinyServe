#include "gpu_tail.h"
#include "smollm3.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

// End-to-end check of the GPU tail inside SmolLM3: the same scripted sequence of cached calls
// runs on the CPU-only model, then on the hybrid model (last N layers + head on the GPU), and
// the logits are compared at every step. The script is built to hit every cache transition:
//   prefill -> decode ... -> truncate + multi-token prefill (device rows must be mirrored back
//   to the host) -> decode -> truncate + one-token suffix -> decode -> reset + new prefill.
// The hybrid run replays the tokens the CPU run picked, so logits stay comparable even if an
// arg-max ever differed. Pass: relative error <= --tol and identical arg-max at every step.

using Clock = std::chrono::steady_clock;

struct Step {
    std::string label;
    std::vector<float> logits;
    double ms = 0.0;
    bool decode = false;  // single-token call
};

static std::size_t argmax(const std::vector<float>& v) {
    return static_cast<std::size_t>(std::max_element(v.begin(), v.end()) - v.begin());
}

class Script {
public:
    // record: pick tokens by arg-max and log them. Replay: use the log.
    Script(SmolLM3<float>& model, std::size_t vocab, std::size_t capacity,
           std::vector<std::size_t>* picks, bool replay)
        : model_{model}, vocab_{vocab}, cache_{model.make_kv_cache(capacity)},
          picks_{picks}, replay_{replay} {}

    std::vector<Step> run(std::size_t prompt_len, std::size_t gen) {
        // 1. prefill
        call("prefill " + std::to_string(prompt_len), ids(0, prompt_len));
        // 2. greedy decode
        for (std::size_t i = 0; i < gen; ++i) decode_next("decode");
        // 3. truncate mid-generation, multi-token suffix: needs the device-made rows on host
        const std::size_t keep = prompt_len + gen / 2;
        cache_.truncate(keep);
        call("truncate->" + std::to_string(keep) + " + prefill 5", ids(200, 5));
        for (int i = 0; i < 3; ++i) decode_next("decode");
        // 4. truncate, suffix of exactly one token (a decode-shaped "prefill")
        cache_.truncate(prompt_len + 1);
        call("truncate->" + std::to_string(prompt_len + 1) + " + suffix of 1", ids(300, 1));
        for (int i = 0; i < 2; ++i) decode_next("decode");
        // 5. reset and start over
        cache_.reset();
        call("reset + prefill 7", ids(400, 7));
        decode_next("decode");
        return std::move(steps_);
    }

private:
    std::vector<uint32_t> ids(std::size_t salt, std::size_t n) const {
        std::vector<uint32_t> out(n);
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = static_cast<uint32_t>((1000 + 7919 * (salt + i)) % vocab_);
        }
        return out;
    }

    void call(const std::string& label, const std::vector<uint32_t>& tokens) {
        const auto t0 = Clock::now();
        Mat<float> logits = model_.next_token_logits(tokens, cache_);
        Step step;
        step.ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        step.label = label;
        step.decode = tokens.size() == 1;
        step.logits.resize(logits.cols());
        for (std::size_t c = 0; c < logits.cols(); ++c) step.logits[c] = logits(0, c);
        steps_.push_back(std::move(step));
    }

    // Feed the next token: the arg-max of the previous logits (recorded or replayed).
    void decode_next(const std::string& label) {
        std::size_t token;
        if (replay_) {
            token = (*picks_).at(pick_index_++);
        } else {
            token = argmax(steps_.back().logits);
            picks_->push_back(token);
        }
        call(label, {static_cast<uint32_t>(token)});
    }

    SmolLM3<float>& model_;
    std::size_t vocab_;
    KVCache<float> cache_;
    std::vector<std::size_t>* picks_;
    bool replay_;
    std::size_t pick_index_ = 0;
    std::vector<Step> steps_;
};

static double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int main(int argc, char* argv[]) {
    try {
        std::filesystem::path model_root = "smollm3";
        std::size_t gpu_layers = 5;
        bool gpu_head = true;
        std::size_t prompt_len = 12;
        std::size_t gen = 10;
        std::size_t threads = mat_detail::default_thread_count();
        double tol = 1e-3;

        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            const bool has_value = i + 1 < argc;
            if (arg == "--gpu-layers" && has_value) gpu_layers = std::stoull(argv[++i]);
            else if (arg == "--no-gpu-head") gpu_head = false;
            else if (arg == "--prompt" && has_value) prompt_len = std::stoull(argv[++i]);
            else if (arg == "--gen" && has_value) gen = std::stoull(argv[++i]);
            else if (arg == "--threads" && has_value) threads = std::stoull(argv[++i]);
            else if (arg == "--tol" && has_value) tol = std::stod(argv[++i]);
            else if (!arg.starts_with("--")) model_root = arg;
            else {
                std::fprintf(stderr, "usage: gpu_tail_check [model_dir] [--gpu-layers N] "
                                     "[--no-gpu-head] [--prompt P] [--gen G] [--threads T] "
                                     "[--tol X]\n");
                return 2;
            }
        }
        if (prompt_len < 2 || gen < 2) {
            std::fprintf(stderr, "--prompt and --gen must be at least 2\n");
            return 2;
        }

        SmolLM3<float>::Config config;
        std::printf("Loading model from '%s' (%zu threads)...\n", model_root.string().c_str(),
                    threads);
        SmolLM3<float> model(model_root, config, {}, threads);
        const std::size_t vocab = model.embed_weights().out_features();
        const std::size_t capacity = prompt_len + gen + 32;

        std::vector<std::size_t> picks;
        std::printf("Running the script on the CPU only...\n");
        std::vector<Step> ref = Script(model, vocab, capacity, &picks, false).run(prompt_len, gen);

        auto upload_start = Clock::now();
        model.set_tail(gpu::make_gpu_tail(model, gpu_layers, gpu_head, capacity));
        std::printf("%s (uploaded in %.0f ms)\nRunning the same script on the hybrid model...\n\n",
                    model.tail()->description().c_str(),
                    std::chrono::duration<double, std::milli>(Clock::now() - upload_start)
                        .count());
        std::vector<Step> hyb = Script(model, vocab, capacity, &picks, true).run(prompt_len, gen);

        std::printf("%-34s  %11s  %7s  %9s  %9s\n", "step", "logits err", "top-1", "cpu ms",
                    "hybrid ms");
        bool all_ok = ref.size() == hyb.size();
        double worst = 0.0;
        std::vector<double> cpu_decode, hyb_decode;
        for (std::size_t i = 0; i < std::min(ref.size(), hyb.size()); ++i) {
            double max_diff = 0.0, max_ref = 0.0;
            for (std::size_t v = 0; v < vocab; ++v) {
                max_diff = std::max(max_diff, std::fabs(static_cast<double>(hyb[i].logits[v]) -
                                                        ref[i].logits[v]));
                max_ref = std::max(max_ref, std::fabs(static_cast<double>(ref[i].logits[v])));
            }
            const double err = max_ref == 0.0 ? max_diff : max_diff / max_ref;
            const bool same = argmax(ref[i].logits) == argmax(hyb[i].logits);
            worst = std::max(worst, err);
            all_ok = all_ok && same && err <= tol;
            if (ref[i].decode) {
                cpu_decode.push_back(ref[i].ms);
                hyb_decode.push_back(hyb[i].ms);
            }
            std::printf("%-34s  %11.3e  %7s  %9.1f  %9.1f\n", ref[i].label.c_str(), err,
                        same ? "same" : "DIFF", ref[i].ms, hyb[i].ms);
        }

        std::printf("\ndecode-shaped calls, median: CPU %.1f ms, hybrid %.1f ms\n",
                    median(cpu_decode), median(hyb_decode));
        std::printf("worst relative error %.3e (tolerance %.1e): %s\n", worst, tol,
                    all_ok ? "PASS" : "FAIL");
        return all_ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "gpu_tail_check: %s\n", e.what());
        return 1;
    }
}
