#pragma once

#include "decoder.h"
#include "smoll_agent.h"
#include "smollm3.h"

#include <chrono>
#include <algorithm>
#include <cstdint>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

enum class StopReason { MaxTokens, EndOfTurn, Cancelled };

template <typename T>
struct GenerationResult {
    std::vector<uint32_t> ids;  // generated tokens, without the closing <|im_end|>
    std::string text;           // ids decoded in one go
    StopReason stop = StopReason::MaxTokens;
    std::size_t prompt_tokens = 0;
    std::size_t reused_tokens = 0;  // prompt prefix already in the KV cache (not recomputed)
    double prefill_ms = 0.0;    // new prompt tokens -> first token
    double decode_ms = 0.0;     // every following token
};

// A SmollAgent bound to a SmolLM3 model: renders the history, tokenizes it and generates.
// Generation is two phases over a KV cache: prefill (whole prompt, fills the cache, gives the
// first token) then decode (one token per step, attending over the cache).
template <typename T>
class SmollSession {
public:
    explicit SmollSession(SmolLM3<T>& model, SmollAgent agent = SmollAgent{})
        : model_{model}, agent_{std::move(agent)} {}

    // Conversation state (system prompt, tools, thinking, messages) lives on the agent.
    SmollAgent& agent() { return agent_; }
    const SmollAgent& agent() const { return agent_; }

    // History rendered up to and including the assistant generation prompt.
    std::string prompt() const { return agent_.prompt(true); }

    std::vector<uint32_t> prompt_ids() const { return model_.tokenizer().encode(prompt()); }

    // Prefill the whole rendered history -> last-position logits (1, vocab_size).
    Mat<T> prefill_logits() { return model_.next_token_logits(prompt_ids()); }

    // Prefill the whole rendered history and pick the first assistant token.
    // DECISION: no KV cache yet, so this recomputes from position 0 on every call.
    DecodedToken<T> first_token(const Decoder<T>& decoder) {
        return decoder.decode_next(prefill_logits(), model_.tokenizer());
    }

    // Generate the assistant turn: stops after max_new_tokens, when the model closes its turn
    // with <|im_end|> (not included in the result), or when on_token returns false.
    // on_token sees each token as it is picked, including the closing <|im_end|>; its text can be a partial UTF-8 sequence,
    // result.text is the reliable decoding.
    //
    // The KV cache lives in the session and survives between calls. The cache remembers which
    // token ids it holds; each call re-renders the whole history, keeps the longest prefix
    // shared with the cached ids and prefills only what follows. A conversation that grows turn
    // by turn therefore only pays for the new messages. Editing earlier history (or switching
    // /think) just shortens the shared prefix, so correctness never depends on how it changed.
    GenerationResult<T> generate(
        const Decoder<T>& decoder, std::size_t max_new_tokens = 20,
        const std::function<bool(const DecodedToken<T>&)>& on_token = {}) {
        using Clock = std::chrono::steady_clock;
        const auto ms = [](Clock::time_point a, Clock::time_point b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };

        GenerationResult<T> result;
        if (max_new_tokens == 0) return result;

        const Tokenizer& tokenizer = model_.tokenizer();
        const uint32_t end_of_turn = tokenizer.added_token_id(chat::kImEnd);
        const std::vector<uint32_t> prompt = prompt_ids();
        result.prompt_tokens = prompt.size();

        if (!cache_) cache_.emplace(model_.make_kv_cache(context_tokens_));
        KVCache<T>& cache = *cache_;
        if (prompt.size() >= cache.capacity()) {
            throw std::length_error("context full: " + std::to_string(prompt.size()) +
                                    " prompt tokens for a context of " +
                                    std::to_string(cache.capacity()));
        }
        // Every generated token but the last one is fed back, so the cache never holds more.
        max_new_tokens = std::min(max_new_tokens, cache.capacity() - prompt.size());

        // Reuse the shared prefix, but always leave at least one token to compute the logits.
        std::size_t reuse = 0;
        while (reuse < cached_ids_.size() && reuse < prompt.size() &&
               cached_ids_[reuse] == prompt[reuse]) {
            ++reuse;
        }
        reuse = std::min(reuse, prompt.size() - 1);
        cache.truncate(reuse);
        cached_ids_.resize(reuse);
        result.reused_tokens = reuse;

        auto t0 = Clock::now();
        Mat<T> logits = model_.next_token_logits(
            std::vector<uint32_t>(prompt.begin() + reuse, prompt.end()), cache);  // prefill
        cached_ids_ = prompt;
        auto t1 = Clock::now();
        result.prefill_ms = ms(t0, t1);

        while (true) {
            DecodedToken<T> token = decoder.decode_next(logits, tokenizer);
            if (token.token_id == end_of_turn) {
                if (on_token) on_token(token);  // shown to the caller, but not part of result.ids
                result.stop = StopReason::EndOfTurn;
                break;
            }
            result.ids.push_back(token.token_id);
            if (on_token && !on_token(token)) {
                result.stop = StopReason::Cancelled;
                break;
            }
            if (result.ids.size() >= max_new_tokens) break;
            logits = model_.next_token_logits({token.token_id}, cache);  // decode
            cached_ids_.push_back(token.token_id);
        }
        result.decode_ms = ms(t1, Clock::now());
        result.text = tokenizer.decode(result.ids);
        return result;
    }

    // Maximum conversation length (prompt + generated) the cache can hold. Takes effect when the
    // cache is next created, i.e. call it before the first generate() or after drop_cache().
    void set_context_tokens(std::size_t n) { context_tokens_ = n; }
    std::size_t context_tokens() const { return context_tokens_; }

    // Free the cache; the next generate() starts from scratch.
    void drop_cache() {
        cache_.reset();
        cached_ids_.clear();
    }

private:
    SmolLM3<T>& model_;
    SmollAgent agent_;
    std::size_t context_tokens_ = 4096;
    std::optional<KVCache<T>> cache_;
    std::vector<uint32_t> cached_ids_;  // token ids whose K/V are in cache_, in order
};
