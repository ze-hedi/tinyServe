#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>
#include "json.hpp"

class Tokenizer {
public:
    explicit Tokenizer(const std::string& tokenizer_json_path) {
        init_regex();
        init_byte_unicode_map();

        // Load tokenizer.json
        std::ifstream f(tokenizer_json_path);
        if (!f.is_open()) {
            throw std::runtime_error("Cannot open: " + tokenizer_json_path);
        }
        nlohmann::json j = nlohmann::json::parse(f);

        // Load vocab: token string -> id
        for (auto& [token, id] : j["model"]["vocab"].items()) {
            uint32_t token_id = id.get<uint32_t>();
            vocab_[token] = token_id;
            id_to_token_[token_id] = token;
        }

        // Load added tokens (<|im_start|>, <think>, ...). They live outside model.vocab and
        // must be matched verbatim in the text before the BPE pre-tokenizer runs.
        if (j.contains("added_tokens")) {
            for (auto& tok : j["added_tokens"]) {
                std::string content = tok["content"].get<std::string>();
                uint32_t token_id = tok["id"].get<uint32_t>();
                added_tokens_.push_back({content, token_id});
                added_to_id_[content] = token_id;
                id_to_added_[token_id] = content;
            }
            // Longest first so that overlapping contents resolve to the longest match.
            std::sort(added_tokens_.begin(), added_tokens_.end(),
                      [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
        }

        // Load merges: ordered list, rank = index
        auto& merges_json = j["model"]["merges"];
        for (std::size_t i = 0; i < merges_json.size(); ++i) {
            std::string first;
            std::string second;
            if (merges_json[i].is_array()) {
                // Recent tokenizer.json files store each merge as [left, right].
                first = merges_json[i][0].get<std::string>();
                second = merges_json[i][1].get<std::string>();
            } else {
                // Older files store each merge as "left right".
                std::string merge_str = merges_json[i].get<std::string>();
                auto sp = merge_str.find(' ');
                if (sp == std::string::npos) {
                    throw std::runtime_error("Invalid tokenizer merge: " + merge_str);
                }
                first = merge_str.substr(0, sp);
                second = merge_str.substr(sp + 1);
            }
            merge_ranks_[{first, second}] = static_cast<uint32_t>(i);
        }
    }

    ~Tokenizer() {
        if (match_data_) pcre2_match_data_free(match_data_);
        if (code_) pcre2_code_free(code_);
    }

    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;

    // Encode: string -> token IDs. Added tokens found in the text map straight to their ids.
    std::vector<uint32_t> encode(const std::string& text) const {
        std::vector<uint32_t> ids;
        std::size_t seg_start = 0;
        std::size_t i = 0;
        while (i < text.size()) {
            if (text[i] == '<') {
                const auto* hit = match_added_token(text, i);
                if (hit) {
                    encode_plain(text.substr(seg_start, i - seg_start), ids);
                    ids.push_back(hit->second);
                    i += hit->first.size();
                    seg_start = i;
                    continue;
                }
            }
            ++i;
        }
        encode_plain(text.substr(seg_start), ids);
        return ids;
    }

    // Decode: token IDs -> string. Added tokens are emitted verbatim unless skip_added is set.
    std::string decode(const std::vector<uint32_t>& ids, bool skip_added = false) const {
        std::string out;
        std::string encoded;
        for (uint32_t id : ids) {
            auto added = id_to_added_.find(id);
            if (added != id_to_added_.end()) {
                out += byte_decode(encoded);
                encoded.clear();
                if (!skip_added) out += added->second;
                continue;
            }
            auto it = id_to_token_.find(id);
            if (it != id_to_token_.end()) {
                encoded += it->second;
            }
        }
        out += byte_decode(encoded);
        return out;
    }

    // Id of an added token such as "<|im_end|>"; throws if the tokenizer doesn't define it.
    uint32_t added_token_id(const std::string& content) const {
        auto it = added_to_id_.find(content);
        if (it == added_to_id_.end()) {
            throw std::runtime_error("Tokenizer: unknown added token '" + content + "'");
        }
        return it->second;
    }

    // Pre-tokenize: regex split
    std::vector<std::string> pre_tokenize(const std::string& text) const {
        std::vector<std::string> chunks;
        auto* subject = reinterpret_cast<PCRE2_SPTR>(text.c_str());
        PCRE2_SIZE len = text.size();
        PCRE2_SIZE offset = 0;

        while (offset < len) {
            int rc = pcre2_match(code_, subject, len, offset, 0,
                                 match_data_, nullptr);
            if (rc < 0) break;

            PCRE2_SIZE* ovector = pcre2_get_ovector_pointer(match_data_);
            PCRE2_SIZE start = ovector[0];
            PCRE2_SIZE end = ovector[1];

            if (end == start) { ++offset; continue; }

            chunks.emplace_back(text.data() + start, end - start);
            offset = end;
        }

        return chunks;
    }

private:
    pcre2_code* code_ = nullptr;
    pcre2_match_data* match_data_ = nullptr;

    // Byte <-> unicode mapping (256 entries)
    std::array<std::string, 256> byte_to_unicode_;
    std::unordered_map<std::string, uint8_t> unicode_to_byte_;

    // Vocab
    std::unordered_map<std::string, uint32_t> vocab_;
    std::unordered_map<uint32_t, std::string> id_to_token_;

    // Added tokens, sorted by descending content length
    std::vector<std::pair<std::string, uint32_t>> added_tokens_;
    std::unordered_map<std::string, uint32_t> added_to_id_;
    std::unordered_map<uint32_t, std::string> id_to_added_;

    const std::pair<std::string, uint32_t>* match_added_token(const std::string& text,
                                                              std::size_t pos) const {
        for (const auto& tok : added_tokens_) {
            if (text.compare(pos, tok.first.size(), tok.first) == 0) return &tok;
        }
        return nullptr;
    }

    // Regex split -> byte-level BPE -> vocab lookup for text with no added tokens in it.
    void encode_plain(const std::string& text, std::vector<uint32_t>& ids) const {
        for (const auto& chunk : pre_tokenize(text)) {
            for (const auto& tok : bpe(byte_encode(chunk))) {
                auto it = vocab_.find(tok);
                if (it != vocab_.end()) {
                    ids.push_back(it->second);
                }
            }
        }
    }

    // BPE merge ranks
    struct PairHash {
        std::size_t operator()(const std::pair<std::string, std::string>& p) const {
            auto h1 = std::hash<std::string>{}(p.first);
            auto h2 = std::hash<std::string>{}(p.second);
            return h1 ^ (h2 << 32) ^ (h2 >> 32);
        }
    };
    std::unordered_map<std::pair<std::string, std::string>, uint32_t, PairHash> merge_ranks_;

    void init_regex() {
        const std::string pattern =
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)"
            "|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+"
            "|\\p{N}{1,3}"
            "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*"
            "|\\s*[\\r\\n]+"
            "|\\s+(?!\\S)"
            "|\\s+";

        int errcode;
        PCRE2_SIZE erroffset;
        code_ = pcre2_compile(
            reinterpret_cast<PCRE2_SPTR>(pattern.c_str()),
            pattern.size(),
            PCRE2_UTF | PCRE2_UCP,
            &errcode, &erroffset, nullptr);

        if (!code_) {
            PCRE2_UCHAR buf[256];
            pcre2_get_error_message(errcode, buf, sizeof(buf));
            throw std::runtime_error(
                "PCRE2 compile error: " + std::string(reinterpret_cast<char*>(buf)));
        }

        match_data_ = pcre2_match_data_create_from_pattern(code_, nullptr);
    }

    // GPT-2 byte-to-unicode map: maps each byte 0-255 to a visible unicode char
    void init_byte_unicode_map() {
        // Printable ranges that map to themselves
        std::vector<int> bs;
        for (int i = 33; i <= 126; ++i) bs.push_back(i);   // '!' to '~'
        for (int i = 161; i <= 172; ++i) bs.push_back(i);   // '¡' to '¬'
        for (int i = 174; i <= 255; ++i) bs.push_back(i);   // '®' to 'ÿ'

        std::vector<int> cs(bs.begin(), bs.end());

        // Non-printable bytes get mapped to U+0100 onwards
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            bool found = false;
            for (int x : bs) { if (x == b) { found = true; break; } }
            if (!found) {
                bs.push_back(b);
                cs.push_back(256 + n);
                ++n;
            }
        }

        // Encode each codepoint as UTF-8
        for (std::size_t i = 0; i < 256; ++i) {
            std::string utf8;
            int cp = cs[i];
            if (cp < 0x80) {
                utf8 += static_cast<char>(cp);
            } else if (cp < 0x800) {
                utf8 += static_cast<char>(0xC0 | (cp >> 6));
                utf8 += static_cast<char>(0x80 | (cp & 0x3F));
            }
            byte_to_unicode_[static_cast<uint8_t>(bs[i])] = utf8;
            unicode_to_byte_[utf8] = static_cast<uint8_t>(bs[i]);
        }
    }

    // Raw bytes -> byte-level unicode string
    std::string byte_encode(const std::string& text) const {
        std::string result;
        for (uint8_t b : text) {
            result += byte_to_unicode_[b];
        }
        return result;
    }

    // Byte-level unicode string -> raw bytes
    std::string byte_decode(const std::string& text) const {
        std::string result;
        std::size_t i = 0;
        while (i < text.size()) {
            // Try 2-byte UTF-8 sequence first
            if (i + 1 < text.size() && (static_cast<uint8_t>(text[i]) & 0xE0) == 0xC0) {
                std::string two(text.data() + i, 2);
                auto it = unicode_to_byte_.find(two);
                if (it != unicode_to_byte_.end()) {
                    result += static_cast<char>(it->second);
                    i += 2;
                    continue;
                }
            }
            // 1-byte
            std::string one(1, text[i]);
            auto it = unicode_to_byte_.find(one);
            if (it != unicode_to_byte_.end()) {
                result += static_cast<char>(it->second);
            }
            ++i;
        }
        return result;
    }

    // Split byte-encoded string into individual unicode characters
    static std::vector<std::string> split_chars(const std::string& s) {
        std::vector<std::string> chars;
        std::size_t i = 0;
        while (i < s.size()) {
            uint8_t c = static_cast<uint8_t>(s[i]);
            std::size_t len = 1;
            if ((c & 0xE0) == 0xC0) len = 2;
            else if ((c & 0xF0) == 0xE0) len = 3;
            else if ((c & 0xF8) == 0xF0) len = 4;
            chars.emplace_back(s.data() + i, std::min(len, s.size() - i));
            i += len;
        }
        return chars;
    }

    // BPE: repeatedly merge the best (lowest rank) pair until no merges remain
    std::vector<std::string> bpe(const std::string& token) const {
        auto word = split_chars(token);
        if (word.size() <= 1) return word;

        while (true) {
            // Find the pair with the lowest rank
            uint32_t best_rank = UINT32_MAX;
            std::size_t best_idx = 0;

            for (std::size_t i = 0; i + 1 < word.size(); ++i) {
                auto it = merge_ranks_.find({word[i], word[i + 1]});
                if (it != merge_ranks_.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_idx = i;
                }
            }

            // No more merges -> done
            if (best_rank == UINT32_MAX) break;

            // Merge the pair
            word[best_idx] += word[best_idx + 1];
            word.erase(word.begin() + best_idx + 1);
        }

        return word;
    }
};
