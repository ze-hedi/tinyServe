#pragma once

#include <cuda_runtime.h>

// Launch wrappers for the single-token (decode) forward pass, float32.
// Plain pointers and ints only, so this header can be included by the host compiler. All
// pointers are device pointers; every launch is asynchronous on `stream`.
//
// Weight matrices use the MMapWeight layout: logical W[K, N] (y = x @ W) with element (k, n) at
// W[n * K + k], i.e. each output column is a contiguous run of K floats.
namespace gpu {

// out = (x / rms(x)) * gamma, rms(x) = sqrt(mean(x^2) + eps).   x, out: (d)
void rmsnorm(const float* x, const float* gamma, float* out, int d, float eps,
             cudaStream_t stream);

// out[n] = sum_k x[k] * W[n*K + k] (+ residual[n] when residual != nullptr).  K % 4 == 0.
void gemv(const float* x, const float* W, const float* residual, float* out, int K, int N,
          cudaStream_t stream);

// out[n] = silu(x . Wgate[n]) * (x . Wup[n])      (SwiGLU gate and up in one pass)
void gemv_swiglu(const float* x, const float* W_gate, const float* W_up, float* out, int K,
                 int N, cudaStream_t stream);

// RoPE (when use_rope) on q in place and on k, then writes the new K row and the V row to
// k_cache/v_cache at row `pos` (row-major (capacity, n_kv_heads * head_dim)).
void rope_append_kv(float* q, const float* k, const float* v, float* k_cache, float* v_cache,
                    int pos, int n_heads, int n_kv_heads, int head_dim, float rope_theta,
                    bool use_rope, cudaStream_t stream);

// Attention of one query token over cache rows [0, visible), grouped-query.
// q, out: (n_heads * head_dim).  Requires head_dim % 4 == 0 and 256 % head_dim == 0.
// Needs visible * 4 bytes of shared memory (throws if the device cannot provide it).
void attention_decode(const float* q, const float* k_cache, const float* v_cache, float* out,
                      int visible, int n_heads, int n_kv_heads, int head_dim,
                      cudaStream_t stream);

}  // namespace gpu
