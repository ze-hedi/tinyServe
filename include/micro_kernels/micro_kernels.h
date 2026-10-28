#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <type_traits>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace micro_kernels {

template <typename T>
inline void mmap_micro_kernel(
        const T* A_ptr, const float* B_ptr, T* C_ptr,
        std::size_t mr, std::size_t nr, std::size_t K_len,
        std::size_t lda, std::size_t ldb, std::size_t ldc) {
    constexpr std::size_t MR = 6;
    constexpr std::size_t NR = 16;
    T acc[MR][NR] = {};
    for (std::size_t i = 0; i < mr; ++i) {
        for (std::size_t j = 0; j < nr; ++j) {
            acc[i][j] = C_ptr[i * ldc + j];
        }
    }
    for (std::size_t i = 0; i < mr; ++i) {
        for (std::size_t k = 0; k < K_len; ++k) {
            const T a = A_ptr[i * lda + k];
            for (std::size_t j = 0; j < nr; ++j) {
                acc[i][j] += a * B_ptr[k * ldb + j];
            }
        }
    }
    for (std::size_t i = 0; i < mr; ++i) {
        for (std::size_t j = 0; j < nr; ++j) {
            C_ptr[i * ldc + j] = acc[i][j];
        }
    }
}

#if defined(__x86_64__) || defined(__i386__)
template <bool Aligned>
__attribute__((target("avx2,fma")))
inline void mmap_micro_kernel_1x16_avx2(
        const float* A, const float* B, float* C, std::size_t K,
        std::size_t ldb) {
    __m256 c0 = _mm256_loadu_ps(C);
    __m256 c1 = _mm256_loadu_ps(C + 8);

    for (std::size_t k = 0; k < K; ++k) {
        __m256 b0;
        __m256 b1;
        if constexpr (Aligned) {
            b0 = _mm256_load_ps(B + k * ldb);
            b1 = _mm256_load_ps(B + k * ldb + 8);
        } else {
            b0 = _mm256_loadu_ps(B + k * ldb);
            b1 = _mm256_loadu_ps(B + k * ldb + 8);
        }
        const __m256 a = _mm256_broadcast_ss(A + k);
        c0 = _mm256_fmadd_ps(a, b0, c0);
        c1 = _mm256_fmadd_ps(a, b1, c1);
    }

    _mm256_storeu_ps(C, c0);
    _mm256_storeu_ps(C + 8, c1);
}

template <bool Aligned>
__attribute__((target("avx2,fma")))
inline void mmap_micro_kernel_6x16_avx2(
        const float* A, const float* B, float* C, std::size_t K,
        std::size_t lda, std::size_t ldb, std::size_t ldc) {
    __m256 c00 = _mm256_loadu_ps(C);
    __m256 c01 = _mm256_loadu_ps(C + 8);
    __m256 c10 = _mm256_loadu_ps(C + ldc);
    __m256 c11 = _mm256_loadu_ps(C + ldc + 8);
    __m256 c20 = _mm256_loadu_ps(C + 2 * ldc);
    __m256 c21 = _mm256_loadu_ps(C + 2 * ldc + 8);
    __m256 c30 = _mm256_loadu_ps(C + 3 * ldc);
    __m256 c31 = _mm256_loadu_ps(C + 3 * ldc + 8);
    __m256 c40 = _mm256_loadu_ps(C + 4 * ldc);
    __m256 c41 = _mm256_loadu_ps(C + 4 * ldc + 8);
    __m256 c50 = _mm256_loadu_ps(C + 5 * ldc);
    __m256 c51 = _mm256_loadu_ps(C + 5 * ldc + 8);

    for (std::size_t k = 0; k < K; ++k) {
        __m256 b0;
        __m256 b1;
        if constexpr (Aligned) {
            b0 = _mm256_load_ps(B + k * ldb);
            b1 = _mm256_load_ps(B + k * ldb + 8);
        } else {
            b0 = _mm256_loadu_ps(B + k * ldb);
            b1 = _mm256_loadu_ps(B + k * ldb + 8);
        }
        c00 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + k), b0, c00);
        c01 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + k), b1, c01);
        c10 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + lda + k), b0, c10);
        c11 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + lda + k), b1, c11);
        c20 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + 2 * lda + k), b0, c20);
        c21 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + 2 * lda + k), b1, c21);
        c30 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + 3 * lda + k), b0, c30);
        c31 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + 3 * lda + k), b1, c31);
        c40 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + 4 * lda + k), b0, c40);
        c41 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + 4 * lda + k), b1, c41);
        c50 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + 5 * lda + k), b0, c50);
        c51 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + 5 * lda + k), b1, c51);
    }

    _mm256_storeu_ps(C, c00);
    _mm256_storeu_ps(C + 8, c01);
    _mm256_storeu_ps(C + ldc, c10);
    _mm256_storeu_ps(C + ldc + 8, c11);
    _mm256_storeu_ps(C + 2 * ldc, c20);
    _mm256_storeu_ps(C + 2 * ldc + 8, c21);
    _mm256_storeu_ps(C + 3 * ldc, c30);
    _mm256_storeu_ps(C + 3 * ldc + 8, c31);
    _mm256_storeu_ps(C + 4 * ldc, c40);
    _mm256_storeu_ps(C + 4 * ldc + 8, c41);
    _mm256_storeu_ps(C + 5 * ldc, c50);
    _mm256_storeu_ps(C + 5 * ldc + 8, c51);
}

// ---- Column-major weight kernels: C(i, j) += dot(A row i, W column j) ----
//
// A rows and W columns are both contiguous over k, so every operand is a plain 8-float load:
// acc(r, c) holds 8 partial sums of the same C(r, c), one per k lane, reduced by a single
// horizontal sum after the kc loop. Per k-step: MR A loads kept in registers, then each W
// column is loaded once and reused for all MR rows. Registers: MR*NR acc + MR A + 1 W
// (2x4 -> 11, 1x8 -> 10 of 16 ymm).

__attribute__((target("avx2,fma")))
inline float hsum256(__m256 v) {
    const __m128 lo = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    const __m128 hi = _mm_movehl_ps(lo, lo);
    const __m128 sum2 = _mm_add_ps(lo, hi);
    return _mm_cvtss_f32(_mm_add_ss(sum2, _mm_shuffle_ps(sum2, sum2, 0x1)));
}

// C[r * ldc + c] += sum_{k < kc} A[r * lda + k] * W[c * ldw + k]  for r < MR, c < NR.
template <std::size_t MR, std::size_t NR>
__attribute__((target("avx2,fma")))
inline void colmajor_dot_kernel_avx2(const float* A, std::size_t lda,
                                     const float* W, std::size_t ldw,
                                     float* C, std::size_t ldc, std::size_t kc) {
    __m256 acc[MR][NR];
#pragma GCC unroll 16
    for (std::size_t r = 0; r < MR; ++r) {
#pragma GCC unroll 16
        for (std::size_t c = 0; c < NR; ++c) acc[r][c] = _mm256_setzero_ps();
    }

    std::size_t k = 0;
    for (; k + 8 <= kc; k += 8) {
        __m256 a[MR];
#pragma GCC unroll 16
        for (std::size_t r = 0; r < MR; ++r) a[r] = _mm256_loadu_ps(A + r * lda + k);
#pragma GCC unroll 16
        for (std::size_t c = 0; c < NR; ++c) {
            const __m256 w = _mm256_loadu_ps(W + c * ldw + k);
#pragma GCC unroll 16
            for (std::size_t r = 0; r < MR; ++r) acc[r][c] = _mm256_fmadd_ps(a[r], w, acc[r][c]);
        }
    }

#pragma GCC unroll 16
    for (std::size_t r = 0; r < MR; ++r) {
#pragma GCC unroll 16
        for (std::size_t c = 0; c < NR; ++c) {
            float sum = hsum256(acc[r][c]);
            for (std::size_t kk = k; kk < kc; ++kk) sum += A[r * lda + kk] * W[c * ldw + kk];
            C[r * ldc + c] += sum;
        }
    }
}

// Decode-only 1x4 kernel: two K blocks per iteration and two independent
// accumulator chains per output. Real model K dimensions are multiples of 16;
// the scalar epilogue keeps arbitrary shapes correct.
__attribute__((target("avx2,fma")))
inline void colmajor_decode4_avx2(const float* A, const float* W, std::size_t ldw,
                                  float* C, std::size_t K) {
    __m256 acc0[4];
    __m256 acc1[4];
#pragma GCC unroll 4
    for (std::size_t c = 0; c < 4; ++c) {
        acc0[c] = _mm256_setzero_ps();
        acc1[c] = _mm256_setzero_ps();
    }

    std::size_t k = 0;
    for (; k + 16 <= K; k += 16) {
        const __m256 a0 = _mm256_loadu_ps(A + k);
        const __m256 a1 = _mm256_loadu_ps(A + k + 8);
#pragma GCC unroll 4
        for (std::size_t c = 0; c < 4; ++c) {
            const float* w = W + c * ldw + k;
            acc0[c] = _mm256_fmadd_ps(a0, _mm256_loadu_ps(w), acc0[c]);
            acc1[c] = _mm256_fmadd_ps(a1, _mm256_loadu_ps(w + 8), acc1[c]);
        }
    }

#pragma GCC unroll 4
    for (std::size_t c = 0; c < 4; ++c) {
        float sum = hsum256(_mm256_add_ps(acc0[c], acc1[c]));
        for (std::size_t kk = k; kk < K; ++kk) sum += A[kk] * W[c * ldw + kk];
        C[c] += sum;
    }
}

inline bool has_avx2_fma() {
    static const bool supported =
        __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    return supported;
}
#endif

template <typename T>
inline void mmap_matmul_tile(
        const T* A, const float* B, T* C, std::size_t K, std::size_t N,
        std::size_t i0, std::size_t i_end, std::size_t j0,
        std::size_t j_end, std::size_t k0, std::size_t k_end,
        bool use_avx2, bool aligned_b) {
    constexpr std::size_t MR = 6;
    constexpr std::size_t NR = 16;
    const std::size_t k_len = k_end - k0;
    for (std::size_t i = i0; i < i_end; i += MR) {
        const std::size_t mr = std::min(MR, i_end - i);
        for (std::size_t j = j0; j < j_end; j += NR) {
            const std::size_t nr = std::min(NR, j_end - j);
#if defined(__x86_64__) || defined(__i386__)
            if constexpr (std::is_same_v<T, float>) {
                if (use_avx2 && mr == 1 && nr == NR) {
                    if (aligned_b) {
                        mmap_micro_kernel_1x16_avx2<true>(
                            A + i * K + k0, B + k0 * N + j, C + i * N + j,
                            k_len, N);
                    } else {
                        mmap_micro_kernel_1x16_avx2<false>(
                            A + i * K + k0, B + k0 * N + j, C + i * N + j,
                            k_len, N);
                    }
                    continue;
                }
                if (use_avx2 && mr == MR && nr == NR) {
                    if (aligned_b) {
                        mmap_micro_kernel_6x16_avx2<true>(
                            A + i * K + k0, B + k0 * N + j, C + i * N + j,
                            k_len, K, N, N);
                    } else {
                        mmap_micro_kernel_6x16_avx2<false>(
                            A + i * K + k0, B + k0 * N + j, C + i * N + j,
                            k_len, K, N, N);
                    }
                    continue;
                }
            }
#else
            (void)use_avx2;
            (void)aligned_b;
#endif
            mmap_micro_kernel(A + i * K + k0, B + k0 * N + j, C + i * N + j,
                              mr, nr, k_len, K, N, N);
        }
    }
}

} // namespace micro_kernels
