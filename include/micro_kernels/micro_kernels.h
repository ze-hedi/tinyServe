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
