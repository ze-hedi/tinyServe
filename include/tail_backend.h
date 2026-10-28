#pragma once

#include <cstddef>
#include <string>

// The last transformer layers of the model, optionally followed by the final norm and lm_head,
// running on another device (the GPU). No device types appear here, so SmolLM3 and the CPU-only
// build do not depend on any GPU library.
//
// The tail owns the K/V rows of its layers on the device. SmolLM3 moves rows between that
// copy and the host KVCache with upload_kv/download_kv (see SmolLM3::next_token_logits).
// Hidden states and logits are plain host float arrays; layer indices are global (0-based).
// float32 only.
class TailBackend {
public:
    virtual ~TailBackend() = default;

    // Global index of the first layer the tail runs; it runs every layer from there on.
    virtual std::size_t first_layer() const = 0;
    virtual bool has_head() const = 0;
    // Tokens the device KV cache can hold (the host cache must not be larger).
    virtual std::size_t kv_capacity() const = 0;
    virtual std::string description() const = 0;

    // Rows [row0, row0 + rows) of one tail layer; k/v are (rows, kv_dim) row-major host arrays.
    virtual void upload_kv(std::size_t layer, std::size_t row0, std::size_t rows, const float* k,
                           const float* v) = 0;
    virtual void download_kv(std::size_t layer, std::size_t row0, std::size_t rows, float* k,
                             float* v) = 0;

    // One token through the tail layers at absolute position `pos`: rows [0, pos) must already
    // be in the device cache; row `pos` is written. hidden_in/hidden_out: (d_model).
    virtual void decode(const float* hidden_in, float* hidden_out, std::size_t pos) = 0;

    // decode() followed by the head, without bringing the hidden state back. Needs has_head().
    virtual void decode_logits(const float* hidden_in, std::size_t pos, float* logits_out) = 0;

    // Final norm + lm_head of one hidden row (d_model) -> logits (vocab). Needs has_head().
    virtual void head_logits(const float* hidden, float* logits_out) = 0;
};
