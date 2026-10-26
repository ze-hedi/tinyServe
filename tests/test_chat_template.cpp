#include "chat_template.h"
#include "json.hpp"
#include "tokenizer.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

// Golden fixtures are rendered by the real HF template (tools/gen_chat_golden.py).
// Strings contain real French/Arabic text: never normalize them.

namespace {

const std::filesystem::path kRoot = TINYSERVE_SOURCE_DIR;

nlohmann::json load_golden() {
    std::ifstream f(kRoot / "tests/fixtures/chat_template_golden.json");
    EXPECT_TRUE(f.is_open());
    return nlohmann::json::parse(f);
}

const Tokenizer& tokenizer() {
    static Tokenizer tok((kRoot / "smollm3/tokenizer/tokenizer.json").string());
    return tok;
}

chat::Role parse_role(const std::string& r) {
    if (r == "user") return chat::Role::User;
    if (r == "assistant") return chat::Role::Assistant;
    return chat::Role::Tool;
}

}  // namespace

TEST(ChatTemplate, MatchesHuggingFaceGolden) {
    const auto golden = load_golden();
    const std::string date = golden["date"];

    for (auto& [name, c] : golden["cases"].items()) {
        chat::ChatTemplate tmpl([&] { return date; });
        tmpl.set_thinking(c["thinking"].get<bool>());
        if (!c["system"].is_null()) tmpl.set_system_prompt(c["system"].get<std::string>());
        tmpl.set_tools(c["tools"].get<std::vector<std::string>>());

        std::vector<chat::Message> msgs;
        for (auto& m : c["messages"]) {
            msgs.push_back({parse_role(m["role"]), m["content"].get<std::string>()});
        }
        const std::string rendered = tmpl.render(msgs, true);
        EXPECT_EQ(rendered, c["prompt"].get<std::string>()) << "case: " << name;

        const auto ids = tokenizer().encode(rendered);
        EXPECT_EQ(ids, c["ids"].get<std::vector<uint32_t>>()) << "case: " << name;
    }
}

TEST(Tokenizer, EncodesAddedTokensAsSingleIds) {
    const auto ids = tokenizer().encode("<|im_start|>user\nHi<|im_end|>\n");
    ASSERT_GE(ids.size(), 3u);
    EXPECT_EQ(ids.front(), 128011u);
    EXPECT_EQ(ids[ids.size() - 2], 128012u);
    EXPECT_EQ(tokenizer().added_token_id("<|im_end|>"), 128012u);
}

TEST(Tokenizer, DecodeRoundTripsAddedTokens) {
    const std::string text = "<|im_start|>user\nBonjour — مرحبا<|im_end|>\n<think>\n\n</think>\n";
    EXPECT_EQ(tokenizer().decode(tokenizer().encode(text)), text);
    EXPECT_EQ(tokenizer().decode({128012}, /*skip_added=*/true), "");
}
