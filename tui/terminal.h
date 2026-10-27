#pragma once

#include <poll.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

#include <cstdio>
#include <stdexcept>
#include <string>

// Minimal terminal control with ANSI escapes. The screen is split in two:
//   rows 1..chat_rows            the chat area, a scroll region: text written there scrolls
//   rows chat_rows+1..rows       the input box, drawn by InputBox
// The chat cursor is parked with ESC 7 / ESC 8 so the input box can move the cursor freely.

inline void out(const std::string& s) {
    std::fwrite(s.data(), 1, s.size(), stdout);
    std::fflush(stdout);
}

// Bytes waiting on stdin, waiting at most timeout_ms (-1 = forever). "" means nothing arrived.
// End of input is reported as Ctrl-D.
inline std::string read_input(int timeout_ms) {
    pollfd fd{STDIN_FILENO, POLLIN, 0};
    if (poll(&fd, 1, timeout_ms) <= 0) return "";
    char buf[256];
    const ssize_t n = read(STDIN_FILENO, buf, sizeof(buf));
    return n > 0 ? std::string(buf, static_cast<std::size_t>(n)) : std::string("\x04");
}

class Terminal {
public:
    int rows = 24;
    int cols = 80;
    int chat_rows = 0;

    explicit Terminal(int reserved_rows) {
        if (!isatty(STDIN_FILENO) || !isatty(STDOUT_FILENO)) {
            throw std::runtime_error("tui_chat needs an interactive terminal");
        }
        tcgetattr(STDIN_FILENO, &saved_);
        termios raw = saved_;
        raw.c_lflag &= ~static_cast<tcflag_t>(ICANON | ECHO | ISIG);  // keys arrive one by one
        raw.c_iflag &= ~static_cast<tcflag_t>(ICRNL | IXON);           // Enter is \r, Ctrl-J is \n
        tcsetattr(STDIN_FILENO, TCSANOW, &raw);

        winsize ws{};
        if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_row > 0) {
            rows = ws.ws_row;
            cols = ws.ws_col;
        }
        chat_rows = rows - reserved_rows;
        // The chat starts on the first row and scrolls only once the area is full.
        out("\x1b[2J\x1b[1;" + std::to_string(chat_rows) + "r\x1b[1;1H\x1b" "7");
    }

    ~Terminal() {
        out("\x1b[r\x1b[" + std::to_string(rows) + ";1H\r\n");
        tcsetattr(STDIN_FILENO, TCSANOW, &saved_);
    }

    Terminal(const Terminal&) = delete;
    Terminal& operator=(const Terminal&) = delete;

    // Everything written between chat_begin() and chat_end() lands in the chat area.
    void chat_begin() const { out("\x1b" "8"); }
    void chat_end() const { out("\x1b" "7"); }

private:
    termios saved_{};
};
