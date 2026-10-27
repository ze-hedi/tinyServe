#include "decoder.h"
#include "input_box.h"
#include "smoll_session.h"
#include "smollm3.h"
#include "terminal.h"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <string>

// Chat on the terminal with the input box fixed at the bottom. Same conversation logic as
// bench_generate: every turn joins the session history and the KV cache is kept.
// The model runs on the main thread; the token callback polls the keyboard to stop the answer.

static const std::string kDim = "\x1b[2m";
static const std::string kBold = "\x1b[1m";
static const std::string kReset = "\x1b[0m";

// Length of the longest prefix of s that does not end in the middle of a UTF-8 character
// (same helper as bench_generate: tokens can split a character).
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

// Prints the streamed answer: only complete UTF-8 characters, and <think>...</think> in dim.
class Streamer {
public:
    void push(const std::string& text) {
        pending_ += text;
        flush(false);
    }

    void finish() {
        flush(true);
        out(kReset);
    }

private:
    static constexpr const char* kOpen = "<think>";
    static constexpr const char* kClose = "</think>";
    std::string pending_;

    void flush(bool all) {
        while (!pending_.empty()) {
            const std::size_t open = pending_.find(kOpen);
            const std::size_t close = pending_.find(kClose);
            if (open == std::string::npos && close == std::string::npos) break;
            const bool is_open = open < close;
            const std::size_t pos = is_open ? open : close;
            out(pending_.substr(0, pos) + (is_open ? kDim : kReset));
            pending_.erase(0, pos + std::string(is_open ? kOpen : kClose).size());
        }
        std::size_t n = pending_.size();
        if (!all) {
            n = complete_utf8_prefix(pending_);
            for (const std::string& tag : {std::string(kOpen), std::string(kClose)}) {
                // Keep back the end of the text if it could be the start of a tag.
                for (std::size_t k = 1; k < tag.size() && k <= pending_.size(); ++k) {
                    if (pending_.ends_with(tag.substr(0, k))) n = std::min(n, pending_.size() - k);
                }
            }
        }
        out(pending_.substr(0, n));
        pending_.erase(0, n);
    }
};

// A dim line in the chat area, for notices.
static void chat_note(const Terminal& term, const std::string& text) {
    term.chat_begin();
    out("\n" + kDim + text + kReset + "\n");
    term.chat_end();
}

// One user turn: show the message, stream the answer, keep both in the history.
static void chat_turn(const Terminal& term, SmollSession<float>& session,
                      const Decoder<float>& decoder, const std::string& line) {
    session.agent().add_user(line);

    term.chat_begin();
    out("\n" + kBold + "you>" + kReset + " " + line + "\n\n" + kBold + "assistant>" + kReset + " ");

    Streamer streamer;
    GenerationResult<float> result;
    try {
        result = session.generate(
            decoder, std::numeric_limits<std::size_t>::max(), [&](const DecodedToken<float>& t) {
                streamer.push(t.text);
                const std::string keys = read_input(0);  // Ctrl-C or Esc stops the answer
                return keys.find('\x03') == std::string::npos &&
                       keys.find('\x1b') == std::string::npos;
            });
    } catch (const std::exception& e) {
        streamer.finish();
        session.agent().pop_message();  // the unanswered user message
        out("\n" + kDim + "[error: " + e.what() + " - /reset to start over]" + kReset + "\n");
        term.chat_end();
        return;
    }
    streamer.finish();

    std::string note;
    if (result.stop == StopReason::MaxTokens) note = " [context full]";
    if (result.stop == StopReason::Cancelled) note = " [stopped]";
    out(note + "\n");

    session.agent().add_assistant(result.text);  // without <|im_end|>, the template adds it back

    const std::size_t n = result.ids.size();
    char stats[128];
    std::snprintf(stats, sizeof(stats), "[context %zu tokens, %zu generated", result.prompt_tokens, n);
    std::string stats_line = stats;
    if (n > 1) {
        std::snprintf(stats, sizeof(stats), ", %.2f tok/s", 1000.0 * (n - 1) / result.decode_ms);
        stats_line += stats;
    }
    out(kDim + stats_line + "]" + kReset + "\n");
    term.chat_end();
}

int main(int argc, char* argv[]) {
    try {
        std::string model_root = "smollm3";
        std::string system_prompt = "You are a helpful assistant.";
        bool thinking = true;
        std::size_t threads = mat_detail::hardware_thread_count();
        std::size_t layers = 0;
        std::size_t context = 4096;

        for (int i = 1; i < argc; ++i) {
            std::string arg(argv[i]);
            if (arg == "--system" && i + 1 < argc) {
                system_prompt = argv[++i];
            } else if (arg == "--no_think") {
                thinking = false;
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
        SmolLM3<float> model(model_root, config, {}, threads);

        SmollSession<float> session(model);
        session.set_context_tokens(context);
        session.agent().set_thinking(thinking).set_system_prompt(system_prompt);
        Decoder<float> decoder(DecodingStrategy::Greedy);

        Terminal term(InputBox::kReservedRows);
        InputBox input;
        term.chat_begin();  // no leading blank line: the chat starts on the first row
        out(kDim + "Chat ready (context " + std::to_string(context) +
            " tokens). Commands: /reset /think /no_think /exit" + kReset + "\n");
        term.chat_end();
        input.render(term);

        while (true) {
            const InputBox::Key key = input.feed(read_input(-1));
            if (key == InputBox::Key::Quit) break;
            if (key == InputBox::Key::Submit) {
                const std::string line = input.text();
                input.clear();
                if (line == "/exit") break;
                if (line == "/reset") {
                    session.agent().reset();
                    chat_note(term, "(conversation cleared)");
                } else if (line == "/think" || line == "/no_think") {
                    session.agent().set_thinking(line == "/think");
                    chat_note(term, "(" + line + ")");
                } else if (!line.empty()) {
                    input.render(term);  // show the box emptied while the answer streams
                    chat_turn(term, session, decoder, line);
                }
            }
            input.render(term);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "tui_chat: %s\n", e.what());
        return 1;
    }
}
