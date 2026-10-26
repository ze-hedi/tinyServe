#pragma once

#include "chat_template.h"

#include <string>
#include <utility>
#include <vector>

// Thin wrapper over ChatTemplate that also keeps the message history.
// No model, no tokenizer: it only builds the tagged context text.
class SmollAgent {
public:
    explicit SmollAgent(chat::ChatTemplate tmpl = chat::ChatTemplate{})
        : template_{std::move(tmpl)} {}

    SmollAgent& set_system_prompt(std::string system) {
        template_.set_system_prompt(std::move(system));
        return *this;
    }
    SmollAgent& set_tools(std::vector<std::string> tools) {
        template_.set_tools(std::move(tools));
        return *this;
    }
    SmollAgent& set_thinking(bool enable) {
        template_.set_thinking(enable);
        return *this;
    }

    SmollAgent& add_user(std::string text) { return add(chat::Role::User, std::move(text)); }
    SmollAgent& add_assistant(std::string text) {
        return add(chat::Role::Assistant, std::move(text));
    }
    SmollAgent& add_tool_result(std::string text) {
        return add(chat::Role::Tool, std::move(text));
    }

    void reset() { history_.clear(); }

    // Drop the most recent message (e.g. a user message that could not be answered).
    void pop_message() {
        if (!history_.empty()) history_.pop_back();
    }

    const std::vector<chat::Message>& history() const { return history_; }

    // History rendered with tags. add_generation_prompt appends the open assistant turn
    // that the model continues from.
    std::string prompt(bool add_generation_prompt = true) const {
        return template_.render(history_, add_generation_prompt);
    }

private:
    chat::ChatTemplate template_;
    std::vector<chat::Message> history_;

    SmollAgent& add(chat::Role role, std::string text) {
        history_.push_back({role, std::move(text)});
        return *this;
    }
};
