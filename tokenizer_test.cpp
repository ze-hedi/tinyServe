#include "tokenizer.h"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>

int main() {
    const char* configured_path = std::getenv("TINY_SERVE_TOKENIZER_JSON");
    const std::string tokenizer_json_path = configured_path
        ? configured_path
        : "/home/bouchehdahed/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM3-3B/snapshots/a07cc9a04f16550a088caea529712d1d335b0ac1/tokenizer.json";

    try {
        Tokenizer tokenizer(tokenizer_json_path);
        const std::string paragraph =
            "Reliable software is built through careful measurement, clear interfaces, "
            "and repeated testing. A tokenizer transforms ordinary text into a sequence "
            "of compact identifiers that a language model can process. Performance can "
            "depend on input length, punctuation, whitespace, Unicode characters, and "
            "the number of merges needed to combine byte-level pieces. This sample "
            "includes numbers such as 2026, contractions like we're, and varied "
            "punctuation; it is repeated to provide a useful wall-clock workload. ";

        std::string input;
        for (int i = 0; i < 100; ++i) {
            input += paragraph;
        }

        constexpr int iterations = 5;
        std::size_t total_tokens = 0;
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < iterations; ++i) {
            total_tokens += tokenizer.encode(input).size();
        }
        const auto elapsed = std::chrono::steady_clock::now() - start;
        const auto elapsed_ms =
            std::chrono::duration<double, std::milli>(elapsed).count();

        std::cout << "Input: " << input.size() << " bytes, " << iterations
                  << " iterations, " << total_tokens << " tokens total\n"
                  << "Wall-clock time: " << elapsed_ms << " ms total, "
                  << elapsed_ms / iterations << " ms/iteration\n";
        return total_tokens > 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "Tokenizer benchmark failed: " << e.what() << '\n';
        return 1;
    }
}
