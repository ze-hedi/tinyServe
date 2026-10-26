#include "decoder.h"
#include "smoll_session.h"
#include "smollm3.h"

#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

// Interactive chat on the terminal: type a message, get the streamed answer, repeat. Every turn
// is added to the history and the KV cache is kept, so only the new messages are prefilled.
// Ctrl-C stops the answer being generated (and exits when pressed at the prompt).
// --verify recomputes each turn's last step without cache and checks the cache agrees.

using Clock = std::chrono::steady_clock;

static constexpr std::size_t kMaxNewTokens = 100;

static double ms_since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

static volatile std::sig_atomic_t g_interrupted = 0;
static void on_sigint(int) { g_interrupted = 1; }

// Length of the longest prefix of s that does not end in the middle of a UTF-8 character.
// Tokens can split a character (byte-level BPE), so streaming prints only complete ones.
static std::size_t complete_utf8_prefix(const std::string& s) {
    std::size_t i = s.size();
    std::size_t back = 0;
    while (i > 0 && back < 4 && (static_cast<unsigned char>(s[i - 1]) & 0xC0) == 0x80) {
        --i;
        ++back;
    }
    if (i == 0) return s.size();
    const unsigned char lead = static_cast<unsigned char>(s[i - 1]);
    std::size_t need = 1;
    if ((lead & 0xE0) == 0xC0) need = 2;
    else if ((lead & 0xF0) == 0xE0) need = 3;
    else if ((lead & 0xF8) == 0xF0) need = 4;
    return (back + 1 < need) ? i - 1 : s.size();
}

static void print_help() {
    std::printf(
        "Commands:\n"
        "  /reset           clear the conversation (keeps the system prompt)\n"
        "  /system <text>   set a new system prompt and clear the conversation\n"
        "  /think           reasoning mode (/no_think to turn it off)\n"
        "  /show            print the context as the model sees it, with tags\n"
        "  /help, /exit\n");
}

int main(int argc, char* argv[]) {
    try {
        std::string model_root = "smollm3";
        std::string system_prompt = "You are a helpful assistant.";
        bool thinking = false;
        bool verify = false;
        std::size_t threads = mat_detail::hardware_thread_count();
        std::size_t layers = 0;
        std::size_t context = 4096;

        for (int i = 1; i < argc; ++i) {
            std::string arg(argv[i]);
            if (arg == "--system" && i + 1 < argc) {
                system_prompt = argv[++i];
            } else if (arg == "--think") {
                thinking = true;
            } else if (arg == "--verify") {
                verify = true;
            } else if (arg == "--ctx" && i + 1 < argc) {
                context = std::stoull(argv[++i]);
            } else if (arg == "--threads" && i + 1 < argc) {
                threads = std::stoull(argv[++i]);
            } else if (arg == "--layers" && i + 1 < argc) {
                layers = std::stoull(argv[++i]);
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
        std::printf("Model loaded in %.1f ms\n", ms_since(load_start));

        SmollSession<float> session(model);
        session.set_context_tokens(context);
        session.agent().set_thinking(thinking).set_system_prompt(system_prompt);
        Decoder<float> decoder(DecodingStrategy::Greedy);

        std::printf("Chat ready (%s, context %zu tokens, up to %zu new tokens per answer). "
                    "/help for commands.\n",
                    thinking ? "/think" : "/no_think", context, kMaxNewTokens);

        std::string line;
        while (true) {
            std::printf("\nyou> ");
            std::fflush(stdout);
            if (!std::getline(std::cin, line)) break;  // EOF
            if (line.empty()) continue;

            if (line.starts_with("/")) {
                if (line == "/exit" || line == "/quit") break;
                if (line == "/help") {
                    print_help();
                } else if (line == "/reset") {
                    session.agent().reset();
                    std::printf("(conversation cleared)\n");
                } else if (line.starts_with("/system ")) {
                    session.agent().reset();
                    session.agent().set_system_prompt(line.substr(8));
                    std::printf("(system prompt set, conversation cleared)\n");
                } else if (line == "/think" || line == "/no_think") {
                    session.agent().set_thinking(line == "/think");
                    std::printf("(%s)\n", line.c_str());
                } else if (line == "/show") {
                    std::printf("%s", session.agent().prompt(false).c_str());
                } else {
                    std::printf("unknown command, /help lists them\n");
                }
                continue;
            }

            session.agent().add_user(line);
            const std::vector<uint32_t> prompt_ids = verify ? session.prompt_ids()
                                                            : std::vector<uint32_t>{};

            g_interrupted = 0;
            std::signal(SIGINT, on_sigint);  // only while generating: at the prompt it exits
            std::string pending;             // streamed text not printed yet (partial UTF-8)
            std::printf("\nassistant> ");
            std::fflush(stdout);
            GenerationResult<float> result;
            try {
                result = session.generate(
                    decoder, kMaxNewTokens, [&](const DecodedToken<float>& t) {
                        pending += t.text;
                        const std::size_t n = complete_utf8_prefix(pending);
                        std::fwrite(pending.data(), 1, n, stdout);
                        std::fflush(stdout);
                        pending.erase(0, n);
                        return !g_interrupted;
                    });
            } catch (const std::exception& e) {
                std::signal(SIGINT, SIG_DFL);
                std::printf("\n[error: %s]\n", e.what());
                session.agent().pop_message();  // the unanswered user message
                std::printf("[use /reset to start over]\n");
                continue;
            }
            std::signal(SIGINT, SIG_DFL);
            std::fwrite(pending.data(), 1, pending.size(), stdout);

            switch (result.stop) {
                case StopReason::EndOfTurn: break;
                case StopReason::MaxTokens: std::printf(" [cut at %zu tokens]", kMaxNewTokens); break;
                case StopReason::Cancelled: std::printf(" [interrupted]"); break;
            }
            std::printf("\n");

            // The answer joins the history (without <|im_end|>; the template adds it back).
            session.agent().add_assistant(result.text);

            const std::size_t n = result.ids.size();
            const std::size_t prefilled = result.prompt_tokens - result.reused_tokens;
            std::printf("[context %zu tokens: %zu reused from cache, %zu prefilled in %.0f ms | "
                        "%zu generated",
                        result.prompt_tokens, result.reused_tokens, prefilled, result.prefill_ms, n);
            if (n > 1) {
                std::printf(", %.0f ms/token (%.2f tok/s)", result.decode_ms / (n - 1),
                            1000.0 * (n - 1) / result.decode_ms);
            }
            std::printf("]\n");

            if (verify && n > 0) {
                // Same last step with no cache at all: prompt + every generated token but the last.
                std::vector<uint32_t> full = prompt_ids;
                full.insert(full.end(), result.ids.begin(), result.ids.end() - 1);
                auto start = Clock::now();
                auto token = decoder.decode_next(model.next_token_logits(full), model.tokenizer());
                const bool same = token.token_id == result.ids.back();
                std::printf("[verify: uncached recompute %.0f ms, argmax %u vs cached pick %u -> %s]\n",
                            ms_since(start), token.token_id, result.ids.back(),
                            same ? "MATCH" : "MISMATCH");
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "bench_generate: %s\n", e.what());
        return 1;
    }
}
