#pragma once

#include "terminal.h"

#include <algorithm>
#include <string>
#include <vector>

// Multi-line input fixed at the bottom of the screen. The cursor always sits at the end of the
// text: typing, deleting the last character and adding lines is all a chat needs.
// Only the last kRows lines are shown, and a line wider than the terminal shows its end.
class InputBox {
public:
    static constexpr int kRows = 3;
    static constexpr int kReservedRows = kRows + 1;  // the input rows plus the separator line

    enum class Key { None, Submit, Quit };

    const std::string& text() const { return text_; }
    void clear() { text_.clear(); }

    // Applies the bytes read from the terminal.
    //   Enter sends, Ctrl-J or Alt-Enter add a line, Backspace deletes, Ctrl-C / Ctrl-D quit.
    Key feed(const std::string& bytes) {
        for (std::size_t i = 0; i < bytes.size(); ++i) {
            const unsigned char c = static_cast<unsigned char>(bytes[i]);
            if (c == 0x03 || c == 0x04) return Key::Quit;
            if (c == '\r') return Key::Submit;
            if (c == '\n') {
                text_ += '\n';
            } else if (c == 0x7f || c == 0x08) {
                pop_char();
            } else if (c == 0x1b) {
                const char next = i + 1 < bytes.size() ? bytes[i + 1] : '\0';
                if (next == '\r') {  // Alt-Enter
                    text_ += '\n';
                    ++i;
                } else if (next == '[' || next == 'O') {  // arrows, Home, ...: ignored
                    i += 2;
                    while (i < bytes.size() && (bytes[i] < 0x40 || bytes[i] > 0x7e)) ++i;
                }
            } else if (c >= 0x20) {
                text_ += static_cast<char>(c);
            }
        }
        return Key::None;
    }

    void render(const Terminal& term) const {
        std::vector<std::string> lines{""};
        for (char c : text_) {
            if (c == '\n') lines.emplace_back();
            else lines.back() += c;
        }
        const std::size_t first = lines.size() > kRows ? lines.size() - kRows : 0;

        std::string hint = "Enter: send | Ctrl-J: new line | Ctrl-C: stop answer / quit";
        hint.resize(std::min<std::size_t>(hint.size(), term.cols));
        std::string s = at(term.chat_rows + 1) + "\x1b[2m" + hint + "\x1b[0m";

        int cursor_row = term.chat_rows + 2;
        int cursor_col = 3;
        for (int r = 0; r < kRows; ++r) {
            const std::size_t idx = first + r;
            s += at(term.chat_rows + 2 + r);
            if (idx >= lines.size()) continue;
            const std::string shown = tail_that_fits(lines[idx], term.cols - 3);
            s += (idx == 0 ? "> " : "  ") + shown;
            cursor_row = term.chat_rows + 2 + r;
            cursor_col = 3 + static_cast<int>(char_count(shown));
        }
        s += "\x1b[" + std::to_string(cursor_row) + ";" + std::to_string(cursor_col) + "H";
        out(s);
    }

private:
    std::string text_;

    // Moves to the start of a row and erases it.
    static std::string at(int row) { return "\x1b[" + std::to_string(row) + ";1H\x1b[2K"; }

    static bool is_continuation(char c) { return (static_cast<unsigned char>(c) & 0xC0) == 0x80; }

    static std::size_t char_count(const std::string& s) {
        return static_cast<std::size_t>(std::count_if(
            s.begin(), s.end(), [](char c) { return !is_continuation(c); }));
    }

    // Deletes the last UTF-8 character (not just the last byte).
    void pop_char() {
        while (!text_.empty() && is_continuation(text_.back())) text_.pop_back();
        if (!text_.empty()) text_.pop_back();
    }

    // The end of the line, at most `width` characters. Wide characters count as one.
    static std::string tail_that_fits(const std::string& line, int width) {
        std::size_t start = line.size();
        int n = 0;
        while (start > 0 && n < width) {
            --start;
            while (start > 0 && is_continuation(line[start])) --start;
            ++n;
        }
        return line.substr(start);
    }
};
