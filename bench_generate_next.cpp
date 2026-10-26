#include "decoder.h"
#include "smollm3.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

using Clock = std::chrono::steady_clock;

int main(int argc, char* argv[]) {
    try {
        std::string model_root = "smollm3";
        std::string prompt = "The capital of France is";
        std::size_t threads = mat_detail::hardware_thread_count();
        std::size_t layers = 0;

        for (int i = 1; i < argc; ++i) {
            std::string arg(argv[i]);
            if (arg == "--prompt" && i + 1 < argc) {
                prompt = argv[++i];
            } else if (arg == "--threads" && i + 1 < argc) {
                threads = std::stoull(argv[++i]);
            } else if (arg == "--layers" && i + 1 < argc) {
                layers = std::stoull(argv[++i]);
            } else if (!arg.starts_with("--")) {
                model_root = arg;
            }
        }

        SmolLM3<float>::Config config;
        if (layers != 0) config.n_layers = layers;

        std::printf("Loading model from '%s' (%zu threads)...\n", model_root.c_str(), threads);
        auto load_start = Clock::now();
        SmolLM3<float> model(model_root, config, {}, threads);
        double load_ms = std::chrono::duration<double, std::milli>(Clock::now() - load_start).count();
        std::printf("Model loaded in %.1f ms\n\n", load_ms);

        // Tokenize
        auto tok_start = Clock::now();
        std::vector<uint32_t> ids = model.tokenizer().encode(prompt);
        double tok_ms = std::chrono::duration<double, std::milli>(Clock::now() - tok_start).count();
        std::printf("Prompt: \"%s\"\n", prompt.c_str());
        std::printf("Tokens (%zu): [", ids.size());
        for (std::size_t i = 0; i < ids.size(); ++i) {
            std::printf("%s%u", i ? ", " : "", ids[i]);
        }
        std::printf("]\nTokenization: %.3f ms\n\n", tok_ms);

        // Forward pass
        std::printf("Running generate_next...\n");
        auto fwd_start = Clock::now();
        Mat<float> logits = model.generate_next(prompt);
        double fwd_ms = std::chrono::duration<double, std::milli>(Clock::now() - fwd_start).count();

        // Decode next token (greedy)
        Decoder<float> decoder(DecodingStrategy::Greedy);
        auto token = decoder.decode_next(logits, model.tokenizer());

        std::printf("Forward pass: %.1f ms\n", fwd_ms);
        std::printf("Next token: id=%u logit=%.4f decoded=\"%s\"\n",
                     token.token_id, token.logit, token.text.c_str());

        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "bench_generate_next: %s\n", e.what());
        return 1;
    }
}
