// Dumps the context a SmollAgent builds for a system prompt + user message,
// tags included. Loads only the tokenizer (no model weights), so it runs instantly.
//
//   ./example_chat_context --system "Answer in one short sentence." --user "Hi"
//   ./example_chat_context --user "Hi" --think --ids
#include "smoll_agent.h"
#include "tokenizer.h"

#include <cstdio>
#include <string>
#include <vector>

// Makes newlines visible so the tag boundaries are easy to spot. keep_break also keeps a real
// line break after each marker so a whole prompt stays readable.
static std::string escape(const std::string& s, bool keep_break = false) {
    std::string out;
    for (char c : s) {
        if (c == '\n') out += keep_break ? "\\n\n" : "\\n";
        else out += c;
    }
    return out;
}

int main(int argc, char* argv[]) {
    try {
        std::string tokenizer_path = "smollm3/tokenizer/tokenizer.json";
        std::string system_prompt;
        bool has_system = false;
        std::string user_message = "What is the capital of France?";
        bool thinking = false;
        bool show_ids = false;

        for (int i = 1; i < argc; ++i) {
            std::string arg(argv[i]);
            if (arg == "--system" && i + 1 < argc) {
                system_prompt = argv[++i];
                has_system = true;
            } else if (arg == "--user" && i + 1 < argc) {
                user_message = argv[++i];
            } else if (arg == "--tokenizer" && i + 1 < argc) {
                tokenizer_path = argv[++i];
            } else if (arg == "--think") {
                thinking = true;
            } else if (arg == "--ids") {
                show_ids = true;
            } else {
                std::fprintf(stderr,
                             "usage: %s [--system TEXT] [--user TEXT] [--think] [--ids] "
                             "[--tokenizer PATH]\n",
                             argv[0]);
                return 2;
            }
        }

        SmollAgent agent;
        agent.set_thinking(thinking);
        if (has_system) agent.set_system_prompt(system_prompt);
        agent.add_user(user_message);
        const std::string prompt = agent.prompt(/*add_generation_prompt=*/false);

        std::printf("=== context (each \\n is followed by a real line break) ===\n%s",
                    escape(prompt, true).c_str());
        std::printf("=== end context ===\n");

        Tokenizer tokenizer(tokenizer_path);
        const std::vector<uint32_t> ids = tokenizer.encode(prompt);
        std::printf("\n%zu tokens", ids.size());

        // Tag tokens are single ids >= 128000 (added tokens).
        std::size_t tags = 0;
        for (uint32_t id : ids) tags += (id >= 128000);
        std::printf(" (%zu of them are tag tokens)\n", tags);

        if (show_ids) {
            for (std::size_t i = 0; i < ids.size(); ++i) {
                const std::string text = tokenizer.decode({ids[i]});
                std::printf("%4zu  id=%-6u %s\n", i, ids[i],
                            ids[i] >= 128000 ? text.c_str() : ("\"" + escape(text) + "\"").c_str());
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "example_chat_context: %s\n", e.what());
        return 1;
    }
}
