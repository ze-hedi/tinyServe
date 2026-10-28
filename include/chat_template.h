#pragma once

#include <cctype>
#include <ctime>
#include <functional>
#include <string>
#include <utility>
#include <vector>

// Renders SmolLM3 chat prompts. Mirrors the Jinja chat_template shipped with
// HuggingFaceTB/SmolLM3-3B byte for byte (verified against tests/fixtures/chat_template_golden.json).
//
// Wire format (ChatML-like):
//   <|im_start|>system\n ...metadata, custom instructions, tools... <|im_end|>\n
//   <|im_start|>user\n...<|im_end|>\n
//   <|im_start|>assistant\n[<think>\n\n</think>\n]...<|im_end|>\n
// Tool results are sent back as a *user* turn.

namespace chat {

enum class Role { User, Assistant, Tool };

struct Message {
    Role role;
    std::string content;
};

struct Tool {
    std::string name;
    std::string definition;
    std::string prompt_snippet;
};

inline constexpr const char* kImStart = "<|im_start|>";
inline constexpr const char* kImEnd = "<|im_end|>";

class ChatTemplate {
public:
    using DateFn = std::function<std::string()>;

    // date_fn lets tests pin "Today Date"; default is strftime("%d %B %Y") like HF's strftime_now.
    explicit ChatTemplate(DateFn date_fn = default_date) : date_fn_{std::move(date_fn)} {}

    // Free-form instructions placed under "## Custom Instructions". "/think" and "/no_think"
    // inside it override set_thinking(); "/system_override" emits the text raw (no metadata).
    ChatTemplate& set_system_prompt(std::string system) {
        system_ = std::move(system);
        has_system_ = true;
        return *this;
    }

    // definition is the tool already serialized to a single line. HF renders a tool dict with
    // Jinja's `string` filter, i.e. a Python repr ({'type': 'function', ...}); the model saw that
    // form in training, so callers should pass the same form. prompt_snippet is additional guidance
    // rendered under the tool's name in the system prompt.
    ChatTemplate& set_tools(std::vector<Tool> tools) {
        tools_ = std::move(tools);
        return *this;
    }

    ChatTemplate& set_thinking(bool enable) {
        thinking_ = enable;
        return *this;
    }

    bool thinking() const { return thinking_; }

    // One rendered assistant turn, as it appears in history.
    std::string assistant_message(const std::string& content) const {
        return std::string(kImStart) + "assistant\n" + (reasoning_think() ? "" : kEmptyThink) +
               lstrip_newlines(content) + kImEnd + "\n";
    }

    std::string render(const std::vector<Message>& messages, bool add_generation_prompt) const {
        std::string out = render_system();
        const bool think = reasoning_think();
        for (const auto& m : messages) {
            switch (m.role) {
                case Role::User:
                case Role::Tool:
                    out += std::string(kImStart) + "user\n" + m.content + kImEnd + "\n";
                    break;
                case Role::Assistant:
                    out += assistant_message(m.content);
                    break;
            }
        }
        if (add_generation_prompt) {
            out += std::string(kImStart) + "assistant\n" + (think ? "" : kEmptyThink);
        }
        return out;
    }

private:
    static constexpr const char* kEmptyThink = "<think>\n\n</think>\n";

    static constexpr const char* kThinkDefault =
        R"THINK(You are a helpful AI assistant named SmolLM, trained by Hugging Face. Your role as an assistant involves thoroughly exploring questions through a systematic thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracking, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution using the specified format: <think> Thought section </think> Solution section. In the Thought section, detail your reasoning process in steps. Each step should include detailed considerations such as analysing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The Solution section should be logical, accurate, and concise and detail necessary steps needed to reach the conclusion.)THINK";

    DateFn date_fn_;
    std::string system_;
    bool has_system_ = false;
    std::vector<Tool> tools_;
    bool thinking_ = true;

    static std::string default_date() {
        std::time_t t = std::time(nullptr);
        char buf[64];
        std::strftime(buf, sizeof(buf), "%d %B %Y", std::localtime(&t));
        return buf;
    }

    static std::string lstrip_newlines(const std::string& s) {
        std::size_t i = 0;
        while (i < s.size() && s[i] == '\n') ++i;
        return s.substr(i);
    }

    static std::string rstrip(std::string s) {
        while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
        return s;
    }

    static bool contains(const std::string& s, const std::string& needle) {
        return s.find(needle) != std::string::npos;
    }

    static std::string replace_all(std::string s, const std::string& from, const std::string& to) {
        for (std::size_t pos = 0; (pos = s.find(from, pos)) != std::string::npos; pos += to.size()) {
            s.replace(pos, from.size(), to);
        }
        return s;
    }

    // Effective mode after the in-system-prompt override.
    bool reasoning_think() const {
        if (has_system_) {
            if (contains(system_, "/no_think")) return false;
            if (contains(system_, "/think")) return true;
        }
        return thinking_;
    }

    std::string render_system() const {
        const bool think = reasoning_think();
        std::string custom;
        if (has_system_) {
            custom = rstrip(replace_all(replace_all(system_, "/no_think", ""), "/think", ""));
        }

        std::string out = std::string(kImStart) + "system\n";

        if (has_system_ && contains(system_, "/system_override")) {
            out += rstrip(replace_all(custom, "/system_override", ""));
            out += std::string(kImEnd) + "\n";
            return out;
        }

        out += "## Metadata\n\n";
        out += "Knowledge Cutoff Date: June 2025\n";
        out += "Today Date: " + date_fn_() + "\n";
        out += std::string("Reasoning Mode: ") + (think ? "/think" : "/no_think") + "\n\n";
        out += "## Custom Instructions\n\n";
        if (!custom.empty()) {
            out += custom + "\n\n";
        } else if (think) {
            out += std::string(kThinkDefault) + "\n\n";
        } else {
            out += "You are a helpful AI assistant named SmolLM, trained by Hugging Face.\n\n";
        }

        // NOTE: in the upstream template "<|im_end|>\n" is emitted *inside* the tools branch, so
        // without tools the system block is left unterminated. We reproduce that on purpose: the
        // model was trained on this exact rendering.
        if (!tools_.empty()) {
            out += "### Tools\n\n";
            for (const auto& tool : tools_) {
                if (tool.prompt_snippet.empty()) continue;
                out += "#### " + tool.name + "\n\n" + tool.prompt_snippet + "\n\n";
            }
            out += "You may call one or more functions to assist with the user query.\n"
                   "You are provided with function signatures within <tools></tools> XML tags:\n\n"
                   "<tools>\n";
            for (const auto& tool : tools_) out += tool.definition + "\n";
            out += "</tools>\n\n"
                   "For each function call, return a json object with function name and arguments "
                   "within <tool_call></tool_call> XML tags:\n"
                   "<tool_call>\n"
                   "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
                   "</tool_call>";
            out += "\n\n";
            out += std::string(kImEnd) + "\n";
        }
        return out;
    }
};

}  // namespace chat
