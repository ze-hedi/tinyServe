#include "gpu_kernels.h"

#include "cuda_utils.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace gpu {
namespace {

constexpr int kWarp = 32;
constexpr int kGemvWarps = 4;       // output columns per block (one warp per column)
constexpr int kAttnThreads = 256;

__device__ __forceinline__ float warp_sum(float v) {
    for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    }
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
    for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, offset));
    }
    return v;
}

// Block-wide reductions; every thread gets the result. `shared` holds >= 32 floats.
__device__ float block_sum(float v, float* shared) {
    const int lane = threadIdx.x & (kWarp - 1);
    const int warp = threadIdx.x / kWarp;
    const int warps = (blockDim.x + kWarp - 1) / kWarp;
    v = warp_sum(v);
    __syncthreads();  // `shared` may still be read by a previous reduction
    if (lane == 0) shared[warp] = v;
    __syncthreads();
    return warp_sum(lane < warps ? shared[lane] : 0.0f);
}

__device__ float block_max(float v, float* shared) {
    const int lane = threadIdx.x & (kWarp - 1);
    const int warp = threadIdx.x / kWarp;
    const int warps = (blockDim.x + kWarp - 1) / kWarp;
    v = warp_max(v);
    __syncthreads();
    if (lane == 0) shared[warp] = v;
    __syncthreads();
    return warp_max(lane < warps ? shared[lane] : -INFINITY);
}

__device__ __forceinline__ float dot4(const float4 a, const float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

// ---------------------------------------------------------------------------------------------

__global__ void rmsnorm_kernel(const float* __restrict__ x, const float* __restrict__ gamma,
                               float* __restrict__ out, int d, float eps) {
    __shared__ float shared[kWarp];
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) sum_sq += x[i] * x[i];
    sum_sq = block_sum(sum_sq, shared);
    const float rms = sqrtf(sum_sq / static_cast<float>(d) + eps);
    for (int i = threadIdx.x; i < d; i += blockDim.x) out[i] = (x[i] / rms) * gamma[i];
}

// One warp per output column: the column is contiguous, so lanes read consecutive float4s.
// x is reused by every warp (stays in L1/L2); the weights are streamed once.
__global__ void gemv_kernel(const float* __restrict__ x, const float* __restrict__ W,
                            const float* __restrict__ residual, float* __restrict__ out, int K,
                            int N) {
    const int lane = threadIdx.x & (kWarp - 1);
    const int n = blockIdx.x * kGemvWarps + threadIdx.x / kWarp;
    if (n >= N) return;  // whole warp exits together

    const float4* w4 = reinterpret_cast<const float4*>(W + static_cast<size_t>(n) * K);
    const float4* x4 = reinterpret_cast<const float4*>(x);
    float acc = 0.0f;
#pragma unroll 4
    for (int i = lane; i < K / 4; i += kWarp) acc += dot4(__ldcs(w4 + i), __ldg(x4 + i));
    acc = warp_sum(acc);
    if (lane == 0) out[n] = residual ? acc + residual[n] : acc;
}

__global__ void gemv_swiglu_kernel(const float* __restrict__ x, const float* __restrict__ W_gate,
                                   const float* __restrict__ W_up, float* __restrict__ out, int K,
                                   int N) {
    const int lane = threadIdx.x & (kWarp - 1);
    const int n = blockIdx.x * kGemvWarps + threadIdx.x / kWarp;
    if (n >= N) return;

    const float4* g4 = reinterpret_cast<const float4*>(W_gate + static_cast<size_t>(n) * K);
    const float4* u4 = reinterpret_cast<const float4*>(W_up + static_cast<size_t>(n) * K);
    const float4* x4 = reinterpret_cast<const float4*>(x);
    float gate = 0.0f;
    float up = 0.0f;
#pragma unroll 4
    for (int i = lane; i < K / 4; i += kWarp) {
        const float4 xv = __ldg(x4 + i);
        gate += dot4(__ldcs(g4 + i), xv);
        up += dot4(__ldcs(u4 + i), xv);
    }
    gate = warp_sum(gate);
    up = warp_sum(up);
    if (lane == 0) out[n] = (gate / (1.0f + expf(-gate))) * up;
}

// One thread per rotated pair of q (n_heads * half) and of k (n_kv_heads * half), then one
// thread per V element. With use_rope == false the rotation is the identity (cos 1, sin 0).
__global__ void rope_append_kernel(float* __restrict__ q, const float* __restrict__ k,
                                   const float* __restrict__ v, float* __restrict__ k_cache,
                                   float* __restrict__ v_cache, int pos, int n_heads,
                                   int n_kv_heads, int head_dim, float rope_theta,
                                   bool use_rope) {
    const int half = head_dim / 2;
    const int kv_dim = n_kv_heads * head_dim;
    const int pairs = (n_heads + n_kv_heads) * half;
    const int t = blockIdx.x * blockDim.x + threadIdx.x;

    if (t < pairs) {
        const int head = t / half;
        const int j = t % half;
        float cos_t = 1.0f;
        float sin_t = 0.0f;
        if (use_rope) {
            const float frequency =
                powf(rope_theta, static_cast<float>(2 * j) / static_cast<float>(head_dim));
            const float theta = static_cast<float>(pos) / frequency;
            cos_t = cosf(theta);
            sin_t = sinf(theta);
        }
        if (head < n_heads) {
            float* p = q + head * head_dim;
            const float x0 = p[j];
            const float x1 = p[j + half];
            p[j] = x0 * cos_t - x1 * sin_t;
            p[j + half] = x0 * sin_t + x1 * cos_t;
        } else {
            const int kv_head = head - n_heads;
            const float* src = k + kv_head * head_dim;
            float* dst = k_cache + static_cast<size_t>(pos) * kv_dim + kv_head * head_dim;
            const float x0 = src[j];
            const float x1 = src[j + half];
            dst[j] = x0 * cos_t - x1 * sin_t;
            dst[j + half] = x0 * sin_t + x1 * cos_t;
        }
    } else if (t < pairs + kv_dim) {
        const int c = t - pairs;
        v_cache[static_cast<size_t>(pos) * kv_dim + c] = v[c];
    }
}

// One block per query head. scores[] (dynamic shared memory) holds q.k for every visible key.
//   phase 1: warps take keys, lanes split head_dim (float4)      -> scores
//   phase 2: block max, exp, block sum                           -> softmax numerators
//   phase 3: thread = output dim, blockDim/head_dim groups split the keys, then combine
__global__ void attention_decode_kernel(const float* __restrict__ q,
                                        const float* __restrict__ k_cache,
                                        const float* __restrict__ v_cache,
                                        float* __restrict__ out, int visible, int n_heads,
                                        int n_kv_heads, int head_dim) {
    extern __shared__ float scores[];
    __shared__ float reduce[kWarp];
    __shared__ float partial[kAttnThreads];

    const int h = blockIdx.x;
    const int kv_dim = n_kv_heads * head_dim;
    const int kv_off = (h / (n_heads / n_kv_heads)) * head_dim;
    const int tid = threadIdx.x;
    const int lane = tid & (kWarp - 1);
    const int warp = tid / kWarp;
    const int warps = blockDim.x / kWarp;
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    const float4* q4 = reinterpret_cast<const float4*>(q + h * head_dim);
    for (int c = warp; c < visible; c += warps) {
        const float4* k4 =
            reinterpret_cast<const float4*>(k_cache + static_cast<size_t>(c) * kv_dim + kv_off);
        float dot = 0.0f;
        for (int i = lane; i < head_dim / 4; i += kWarp) dot += dot4(q4[i], k4[i]);
        dot = warp_sum(dot);
        if (lane == 0) scores[c] = dot * scale;
    }
    __syncthreads();

    float local_max = -INFINITY;
    for (int c = tid; c < visible; c += blockDim.x) local_max = fmaxf(local_max, scores[c]);
    const float max_score = block_max(local_max, reduce);

    float local_sum = 0.0f;
    for (int c = tid; c < visible; c += blockDim.x) {
        const float e = expf(scores[c] - max_score);
        scores[c] = e;
        local_sum += e;
    }
    const float sum = block_sum(local_sum, reduce);  // also orders the scores[] writes
    __syncthreads();

    const int groups = blockDim.x / head_dim;
    const int d = tid % head_dim;
    const int group = tid / head_dim;
    float acc = 0.0f;
    for (int c = group; c < visible; c += groups) {
        acc += scores[c] * v_cache[static_cast<size_t>(c) * kv_dim + kv_off + d];
    }
    partial[group * head_dim + d] = acc;
    __syncthreads();
    if (group == 0) {
        float total = 0.0f;
        for (int g = 0; g < groups; ++g) total += partial[g * head_dim + d];
        out[h * head_dim + d] = total / sum;
    }
}

void require(bool ok, const char* what) {
    if (!ok) throw std::invalid_argument(std::string("gpu kernels: ") + what);
}

}  // namespace

void rmsnorm(const float* x, const float* gamma, float* out, int d, float eps,
             cudaStream_t stream) {
    require(d > 0, "rmsnorm: d must be positive");
    const int threads = d < 1024 ? ((d + kWarp - 1) / kWarp) * kWarp : 1024;
    rmsnorm_kernel<<<1, threads, 0, stream>>>(x, gamma, out, d, eps);
    CUDA_CHECK(cudaGetLastError());
}

void gemv(const float* x, const float* W, const float* residual, float* out, int K, int N,
          cudaStream_t stream) {
    require(K % 4 == 0, "gemv: K must be a multiple of 4");
    const int blocks = (N + kGemvWarps - 1) / kGemvWarps;
    gemv_kernel<<<blocks, kGemvWarps * kWarp, 0, stream>>>(x, W, residual, out, K, N);
    CUDA_CHECK(cudaGetLastError());
}

void gemv_swiglu(const float* x, const float* W_gate, const float* W_up, float* out, int K,
                 int N, cudaStream_t stream) {
    require(K % 4 == 0, "gemv_swiglu: K must be a multiple of 4");
    const int blocks = (N + kGemvWarps - 1) / kGemvWarps;
    gemv_swiglu_kernel<<<blocks, kGemvWarps * kWarp, 0, stream>>>(x, W_gate, W_up, out, K, N);
    CUDA_CHECK(cudaGetLastError());
}

void rope_append_kv(float* q, const float* k, const float* v, float* k_cache, float* v_cache,
                    int pos, int n_heads, int n_kv_heads, int head_dim, float rope_theta,
                    bool use_rope, cudaStream_t stream) {
    require(head_dim % 2 == 0, "rope_append_kv: head_dim must be even");
    const int threads_needed = (n_heads + n_kv_heads) * (head_dim / 2) + n_kv_heads * head_dim;
    const int block = 256;
    const int blocks = (threads_needed + block - 1) / block;
    rope_append_kernel<<<blocks, block, 0, stream>>>(q, k, v, k_cache, v_cache, pos, n_heads,
                                                     n_kv_heads, head_dim, rope_theta, use_rope);
    CUDA_CHECK(cudaGetLastError());
}

void attention_decode(const float* q, const float* k_cache, const float* v_cache, float* out,
                      int visible, int n_heads, int n_kv_heads, int head_dim,
                      cudaStream_t stream) {
    require(visible > 0, "attention_decode: nothing to attend to");
    require(n_kv_heads > 0 && n_heads % n_kv_heads == 0,
            "attention_decode: n_heads must be divisible by n_kv_heads");
    require(head_dim % 4 == 0 && head_dim <= kAttnThreads && kAttnThreads % head_dim == 0,
            "attention_decode: head_dim must be a multiple of 4 that divides 256");

    const size_t smem = static_cast<size_t>(visible) * sizeof(float);
    if (smem > 48 * 1024) {
        int device = 0;
        int max_optin = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                          device));
        require(smem <= static_cast<size_t>(max_optin),
                "attention_decode: context too long for the shared-memory score buffer");
        CUDA_CHECK(cudaFuncSetAttribute(attention_decode_kernel,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(smem)));
    }
    attention_decode_kernel<<<n_heads, kAttnThreads, smem, stream>>>(
        q, k_cache, v_cache, out, visible, n_heads, n_kv_heads, head_dim);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace gpu
