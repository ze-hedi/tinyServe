#pragma once

#include "mat.h"
#include "tokenizer.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

enum class DecodingStrategy { Greedy };

template <typename T>
struct DecodedToken {
    uint32_t token_id;
    T logit;
    std::string text;
};

template <typename T>
class Decoder {
public:
    explicit Decoder(DecodingStrategy strategy = DecodingStrategy::Greedy)
        : strategy_{strategy} {}

    // logits: (1, vocab_size) — last-position logits from generate_next()
    DecodedToken<T> decode_next(const Mat<T>& logits, const Tokenizer& tokenizer) const {
        if (logits.rows() != 1) {
            throw std::invalid_argument("Decoder: expected logits with 1 row");
        }
        switch (strategy_) {
            case DecodingStrategy::Greedy:
                return greedy(logits, tokenizer);
        }
        __builtin_unreachable();
    }

private:
    DecodingStrategy strategy_;

    DecodedToken<T> greedy(const Mat<T>& logits, const Tokenizer& tokenizer) const {
        std::size_t best_id = 0;
        T best_val = logits(0, 0);
        for (std::size_t c = 1; c < logits.cols(); ++c) {
            if (logits(0, c) > best_val) {
                best_val = logits(0, c);
                best_id = c;
            }
        }
        auto id = static_cast<uint32_t>(best_id);
        std::string text = tokenizer.decode({id});
        return {id, best_val, std::move(text)};
    }
};
