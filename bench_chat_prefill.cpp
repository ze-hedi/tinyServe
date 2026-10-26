#include "decoder.h"
#include "smoll_session.h"
#include "smollm3.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

using Clock = std::chrono::steady_clock;

static double ms_since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

int main(int argc, char* argv[]) {
    try {
        std::string model_root = "smollm3";
        std::string system_prompt;
        bool has_system = false;
        std::string user_message = "What is the capital of France?";
        bool thinking = false;
        bool show_prompt = false;
        bool show_ids = false;
        std::size_t threads = mat_detail::hardware_thread_count();
        std::size_t layers = 0;
        std::size_t repeat = 1;
        std::size_t top_n = 0;

        for (int i = 1; i < argc; ++i) {
            std::string arg(argv[i]);
            if (arg == "--system" && i + 1 < argc) {
                system_prompt = argv[++i];
                has_system = true;
            } else if (arg == "--user" && i + 1 < argc) {
                user_message = argv[++i];
            } else if (arg == "--think") {
                thinking = true;
            } else if (arg == "--show-prompt") {
                show_prompt = true;
            } else if (arg == "--show-ids") {
                show_ids = true;
            } else if (arg == "--threads" && i + 1 < argc) {
                threads = std::stoull(argv[++i]);
            } else if (arg == "--layers" && i + 1 < argc) {
                layers = std::stoull(argv[++i]);
            } else if (arg == "--top" && i + 1 < argc) {
                top_n = std::stoull(argv[++i]);
            } else if (arg == "--repeat" && i + 1 < argc) {
                repeat = std::stoull(argv[++i]);
            } else if (!arg.starts_with("--")) {
                model_root = arg;
            } else {
                std::fprintf(stderr, "unknown or incomplete option: %s\n", arg.c_str());
                return 2;
            }
        }

        SmolLM3<float>::Config config;
        if (layers != 0) config.n_layers = layers;

        std::printf("Loading model from '%s' (%zu threads)...\n", model_root.c_str(), threads);
        auto load_start = Clock::now();
        SmolLM3<float> model(model_root, config, {}, threads);
        std::printf("Model loaded in %.1f ms\n\n", ms_since(load_start));

        SmollSession<float> session(model);
        session.agent().set_thinking(thinking);
        if (has_system) session.agent().set_system_prompt(system_prompt);
        session.agent().add_user(user_message);

        // Template render
        auto render_start = Clock::now();
        const std::string prompt = session.prompt();
        double render_ms = ms_since(render_start);

        // Tokenize
        auto tok_start = Clock::now();
        const std::vector<uint32_t> ids = model.tokenizer().encode(prompt);
        double tok_ms = ms_since(tok_start);

        std::printf("Mode: %s | system prompt: %s\n", thinking ? "/think" : "/no_think",
                    has_system ? "custom" : "default");
        std::printf("User: \"%s\"\n", user_message.c_str());
        if (show_prompt) std::printf("\n--- rendered prompt ---\n%s\n--- end prompt ---\n", prompt.c_str());
        if (show_ids) {
            std::printf("Token ids: [");
            for (std::size_t i = 0; i < ids.size(); ++i) std::printf("%s%u", i ? ", " : "", ids[i]);
            std::printf("]\n");
        }
        std::printf("Prompt tokens: %zu\n", ids.size());
        std::printf("Template render: %.3f ms | Tokenization: %.3f ms\n\n", render_ms, tok_ms);

        // Prefill -> first token (no KV cache: every run recomputes from position 0)
        Decoder<float> decoder(DecodingStrategy::Greedy);
        DecodedToken<float> token{};
        Mat<float> last_logits(1, 1);
        double total_ms = 0.0;
        double best_ms = 0.0;
        for (std::size_t r = 0; r < repeat; ++r) {
            auto fwd_start = Clock::now();
            last_logits = session.prefill_logits();
            token = decoder.decode_next(last_logits, model.tokenizer());
            double ms = ms_since(fwd_start);
            total_ms += ms;
            if (r == 0 || ms < best_ms) best_ms = ms;
            std::printf("Prefill run %zu/%zu: %.1f ms (%.1f tok/s)\n", r + 1, repeat, ms,
                        1000.0 * static_cast<double>(ids.size()) / ms);
        }
        if (repeat > 1) {
            std::printf("Prefill avg: %.1f ms | best: %.1f ms\n",
                        total_ms / static_cast<double>(repeat), best_ms);
        }

        std::printf("\nFirst token: id=%u logit=%.4f decoded=\"%s\"\n", token.token_id,
                    token.logit, token.text.c_str());
        if (top_n > 0) {
            std::vector<std::size_t> idx(last_logits.cols());
            for (std::size_t c = 0; c < idx.size(); ++c) idx[c] = c;
            top_n = std::min(top_n, idx.size());
            std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(top_n),
                              idx.end(), [&](std::size_t a, std::size_t b) {
                                  return last_logits(0, a) > last_logits(0, b);
                              });
            for (std::size_t k = 0; k < top_n; ++k) {
                std::printf("  top%zu: id=%zu logit=%.4f \"%s\"\n", k + 1, idx[k],
                            last_logits(0, idx[k]),
                            model.tokenizer().decode({static_cast<uint32_t>(idx[k])}).c_str());
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "bench_chat_prefill: %s\n", e.what());
        return 1;
    }
}
