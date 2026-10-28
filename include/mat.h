#pragma once

#include "mmap_block_loader.h"
#include "micro_kernels/micro_kernels.h"
#include "thread_pool.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace mat_detail {
inline std::size_t hardware_thread_count() {
    const unsigned int count = std::thread::hardware_concurrency();
    return count == 0 ? 1 : static_cast<std::size_t>(count);
}
// Decode is weight-bandwidth-bound; a small physical-core team outperforms using every SMT/E-core.
inline std::size_t default_thread_count() {
    return std::min<std::size_t>(6, hardware_thread_count());
}
} // namespace mat_detail

template <typename T>
class Mat {
public:
    Mat(std::size_t rows, std::size_t cols)
        : rows_{rows}, cols_{cols}, data_(rows * cols, T{0}) {}

    Mat(std::size_t rows, std::size_t cols, std::vector<T> data)
        : rows_{rows}, cols_{cols}, data_{std::move(data)} {
        if (data_.size() != rows_ * cols_) {
            throw std::invalid_argument("data size does not match dimensions");
        }
    }

    Mat(T* mapped_data, std::size_t rows, std::size_t cols)
        : mmap_data_(mapped_data), rows_{rows}, cols_{cols}{}

    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }

    T& operator()(std::size_t row, std::size_t col) {
        return data_[row * cols_ + col];
    }

    const T& operator()(std::size_t row, std::size_t col) const {
        return data_[row * cols_ + col];
    }

    // A(rows, cols) * v(cols, 1) -> result(rows, 1)
    Mat matvec(const Mat& v) const {
        if (v.cols_ != 1) {
            throw std::invalid_argument("matvec: v must have exactly 1 column");
        }
        if (cols_ != v.rows_) {
            throw std::invalid_argument("matvec: dimension mismatch");
        }

        Mat result(rows_, 1);
        for (std::size_t r = 0; r < rows_; ++r) {
            T sum{0};
            for (std::size_t k = 0; k < cols_; ++k) {
                sum += data_[r * cols_ + k] * v.data_[k];
            }
            result.data_[r] = sum;
        }
        return result;
    }

    // Register-tiled micro-kernel: MR×NR tile of C pinned in local variables
    // C[mr×nr] += A[mr×K_len] * B[K_len×nr]
    // Local vars → compiler maps them to registers, loaded once, stored once
    __attribute__((always_inline))
    static inline void micro_kernel(
            const T* A_ptr, const T* B_ptr, T* C_ptr,
            std::size_t mr, std::size_t nr, std::size_t K_len,
            std::size_t lda, std::size_t ldb, std::size_t ldc) {
        // Pin C tile in local accumulators (up to 6×16 = 96 values)
        // Compiler will use registers for these
        constexpr std::size_t MR = 6;
        constexpr std::size_t NR = 16;
        T acc[MR][NR] = {};

        // Load existing C into accumulators
        for (std::size_t i = 0; i < mr; ++i) {
            for (std::size_t j = 0; j < nr; ++j) {
                acc[i][j] = C_ptr[i * ldc + j];
            }
        }

        // Accumulate A * B into registers — C never touched until the end
        for (std::size_t k = 0; k < K_len; ++k) {
            for (std::size_t i = 0; i < mr; ++i) {
                const T a = A_ptr[i * lda + k];
                for (std::size_t j = 0; j < nr; ++j) {
                    acc[i][j] += a * B_ptr[k * ldb + j];
                }
            }
        }

        // Store accumulators back to C (once!)
        for (std::size_t i = 0; i < mr; ++i) {
            for (std::size_t j = 0; j < nr; ++j) {
                C_ptr[i * ldc + j] = acc[i][j];
            }
        }
    }

    // Cache tile dispatcher: tiles down to MR×NR micro-kernels
    __attribute__((always_inline))
    static inline void matmul_tile(const T* A, const T* B, T* C,
                            std::size_t K, std::size_t N,
                            std::size_t i0, std::size_t i_end, std::size_t j0,
                            std::size_t j_end, std::size_t k0, std::size_t k_end) {
        constexpr std::size_t MR = 6;
        constexpr std::size_t NR = 16;
        const std::size_t K_len = k_end - k0;

        for (std::size_t i = i0; i < i_end; i += MR) {
            const std::size_t mr = std::min(MR, i_end - i);
            for (std::size_t j = j0; j < j_end; j += NR) {
                const std::size_t nr = std::min(NR, j_end - j);
                micro_kernel(
                    A + i * K + k0,
                    B + k0 * N + j,
                    C + i * N + j,
                    mr, nr, K_len, K, N, N);
            }
        }
    }

    // A(m, k) * B(k, n) -> C(m, n)  — 3D tiled for cache locality
    Mat matmul(const Mat& B, ThreadPool& pool) const {
        if (cols_ != B.rows_) {
            throw std::invalid_argument("matmul: dimension mismatch");
        }
        if constexpr (std::is_same_v<T, float>) {
            return matmul_avx_tiled(B.data_.data(), B.cols_, pool);
        }

        const std::size_t M = rows_;
        const std::size_t K = cols_;
        const std::size_t N = B.cols_;
        constexpr std::size_t TI = 64;
        constexpr std::size_t TJ = 64;
        constexpr std::size_t TK = 64;

        Mat result(M, N);
        if (M == 0 || N == 0) {
            return result;
        }
        const T* A_ptr = data_.data();
        const T* B_ptr = B.data_.data();
        T* C_ptr = result.data_.data();

        auto worker = [&](std::size_t row_start, std::size_t row_end) {
            for (std::size_t i0 = row_start; i0 < row_end; i0 += TI) {
                const std::size_t i_end = std::min(i0 + TI, row_end);
                for (std::size_t j0 = 0; j0 < N; j0 += TJ) {
                    const std::size_t j_end = std::min(j0 + TJ, N);
                    for (std::size_t k0 = 0; k0 < K; k0 += TK) {
                        const std::size_t k_end = std::min(k0 + TK, K);
                        matmul_tile(A_ptr, B_ptr, C_ptr, K, N,
                                    i0, i_end, j0, j_end, k0, k_end);
                    }
                }
            }
        };

        const std::size_t worker_count = std::min(pool.size(), M);
        const std::size_t rows_per_worker = (M + worker_count - 1) / worker_count;
        pool.run(worker_count, [&](std::size_t t) {
            const std::size_t row_start = t * rows_per_worker;
            worker(row_start, std::min(row_start + rows_per_worker, M));
        });
        return result;
    }

    // A(m, k) * W(k, n) -> C(m, n), W a column-major MMapWeight: c(i, j) = dot(A row i, W col j),
    // both contiguous over k. M == 1 takes the decode path, M > 1 the tiled prefill path;
    // the scalar path covers non-float T and CPUs without AVX2/FMA.
    Mat matmul(const MMapWeight& W, ThreadPool& pool) const {
        if (cols_ != W.in_features()) {
            throw std::invalid_argument("matmul(MMapWeight): dimension mismatch");
        }
        if (rows_ == 0 || W.out_features() == 0) {
            return Mat(rows_, W.out_features());
        }
#if defined(__x86_64__) || defined(__i386__)
        if constexpr (std::is_same_v<T, float>) {
            if (micro_kernels::has_avx2_fma()) {
                return rows_ == 1 ? matmul_colmajor_decode(W, pool)
                                  : matmul_colmajor_prefill(W, pool);
            }
        }
#endif
        return matmul_colmajor_scalar(W, pool);
    }

    // Threads split N, so each owns disjoint output columns: no reduction, no synchronization.
    Mat matmul_colmajor_scalar(const MMapWeight& W, ThreadPool& pool) const {
        const std::size_t M = rows_;
        const std::size_t K = cols_;
        const std::size_t N = W.out_features();
        Mat result(M, N);
        const T* A_ptr = data_.data();
        T* C_ptr = result.data_.data();
        const std::size_t worker_count = std::min(pool.size(), N);
        pool.run(worker_count, [&](std::size_t t) {
            const std::size_t j_begin = N * t / worker_count;
            const std::size_t j_end = N * (t + 1) / worker_count;
            for (std::size_t i = 0; i < M; ++i) {
                const T* a = A_ptr + i * K;
                for (std::size_t j = j_begin; j < j_end; ++j) {
                    const float* w = W.col(j);
                    T acc{0};
                    for (std::size_t k = 0; k < K; ++k) {
                        acc += a[k] * static_cast<T>(w[k]);
                    }
                    C_ptr[i * N + j] = acc;
                }
            }
        });
        return result;
    }

#if defined(__x86_64__) || defined(__i386__)
    // Decode, v(1, k) * W(k, n): 1x4 micro-kernel over the whole K (one horizontal sum per
    // output). Four weight streams per task measured faster than eight on the memory-bound host.
    // Threads split N in 16-column units (64 B of C) so workers never share a line of the output;
    // units are claimed dynamically so faster P-cores can take more work than E-cores.
    Mat matmul_colmajor_decode(const MMapWeight& W, ThreadPool& pool) const {
        constexpr std::size_t NR = 4;
        constexpr std::size_t UNIT = 16;
        const std::size_t K = cols_;
        const std::size_t N = W.out_features();
        Mat result(1, N);
        const float* a = data_.data();
        float* C_ptr = result.data_.data();

        const std::size_t units = (N + UNIT - 1) / UNIT;
        pool.run(units, [&](std::size_t t) {
            const std::size_t j_begin = t * UNIT;
            const std::size_t j_end = std::min(j_begin + UNIT, N);
            std::size_t j = j_begin;
            for (; j + NR <= j_end; j += NR) {
                micro_kernels::colmajor_decode4_avx2(a, W.col(j), K, C_ptr + j, K);
            }
            for (; j < j_end; ++j) {
                micro_kernels::colmajor_dot_kernel_avx2<1, 1>(a, K, W.col(j), K, C_ptr + j, 0, K);
            }
        });
        return result;
    }

    // Prefill, A(m, k) * W(k, n). Tiles A[TI, KC] x W[KC, TJ]; for each (I, J) tile the K tiles
    // run sequentially in the same thread, accumulating into C, then each tile pair is cut into
    // 2x4 micro-tiles. Threads split M (row blocks), or N when there are too few row blocks to
    // feed the pool; either way each thread owns disjoint C, so no synchronization.
    // DECISION: TI = TJ = 64, KC = 512 -> A and W tiles are 128 KB each (L2-resident on a
    // Raptor Lake P-core), and one horizontal sum per 512 k keeps the reduction cost ~1%.
    Mat matmul_colmajor_prefill(const MMapWeight& W, ThreadPool& pool) const {
        constexpr std::size_t TI = 64;
        constexpr std::size_t TJ = 64;
        constexpr std::size_t KC = 512;
        constexpr std::size_t MR = 2;
        constexpr std::size_t NR = 4;
        const std::size_t M = rows_;
        const std::size_t K = cols_;
        const std::size_t N = W.out_features();
        Mat result(M, N);
        const float* A_ptr = data_.data();
        const float* W_ptr = W.data();
        float* C_ptr = result.data_.data();

        auto micro_tile = [&](std::size_t i, std::size_t mr, std::size_t j, std::size_t nr,
                              std::size_t k0, std::size_t kc) {
            const float* a = A_ptr + i * K + k0;
            const float* w = W_ptr + j * K + k0;
            float* c = C_ptr + i * N + j;
            if (mr == MR && nr == NR) {
                micro_kernels::colmajor_dot_kernel_avx2<MR, NR>(a, K, w, K, c, N, kc);
            } else if (nr == NR) {
                micro_kernels::colmajor_dot_kernel_avx2<1, NR>(a, K, w, K, c, N, kc);
            } else {
                for (std::size_t r = 0; r < mr; ++r) {
                    for (std::size_t col = 0; col < nr; ++col) {
                        micro_kernels::colmajor_dot_kernel_avx2<1, 1>(
                            a + r * K, K, w + col * K, K, c + r * N + col, N, kc);
                    }
                }
            }
        };

        auto block = [&](std::size_t i_begin, std::size_t i_end,
                         std::size_t j_begin, std::size_t j_end) {
            for (std::size_t i0 = i_begin; i0 < i_end; i0 += TI) {
                const std::size_t i1 = std::min(i0 + TI, i_end);
                for (std::size_t j0 = j_begin; j0 < j_end; j0 += TJ) {
                    const std::size_t j1 = std::min(j0 + TJ, j_end);
                    for (std::size_t k0 = 0; k0 < K; k0 += KC) {
                        const std::size_t kc = std::min(KC, K - k0);
                        for (std::size_t i = i0; i < i1; i += MR) {
                            for (std::size_t j = j0; j < j1; j += NR) {
                                micro_tile(i, std::min(MR, i1 - i), j, std::min(NR, j1 - j),
                                           k0, kc);
                            }
                        }
                    }
                }
            }
        };

        const std::size_t row_blocks = (M + MR - 1) / MR;
        const std::size_t col_units = (N + 15) / 16;
        const bool split_columns = row_blocks < pool.size() && col_units > 1;
        const std::size_t task_count = split_columns ? col_units : row_blocks;
        const std::size_t worker_count = std::min(pool.size(), task_count);
        pool.run(worker_count, [&](std::size_t t) {
            const std::size_t task_begin = task_count * t / worker_count;
            const std::size_t task_end = task_count * (t + 1) / worker_count;
            if (split_columns) {
                block(0, M, task_begin * 16, std::min(task_end * 16, N));
            } else {
                block(task_begin * MR, std::min(task_end * MR, M), 0, N);
            }
        });
        return result;
    }
#endif

    // Shared AVX2/FMA + cache-tiled path: A(m, k) * B(k, n), B a raw row-major float buffer
    // (another Mat's storage).
    Mat matmul_avx_tiled(const float* B_ptr, std::size_t N, ThreadPool& pool) const {
        const std::size_t M = rows_;
        const std::size_t K = cols_;
        constexpr std::size_t TI = 64;
        constexpr std::size_t TJ = 64;
        constexpr std::size_t TK = 64;
        constexpr std::size_t MR = 6;
        constexpr std::size_t NR = 16;

        Mat result(M, N);
        if (M == 0 || N == 0) {
            return result;
        }
        const T* A_ptr = data_.data();
        T* C_ptr = result.data_.data();

        bool use_avx2 = false;
#if defined(__x86_64__) || defined(__i386__)
        if constexpr (std::is_same_v<T, float>) {
            use_avx2 = micro_kernels::has_avx2_fma();
        }
#endif
        const bool aligned_b = reinterpret_cast<std::uintptr_t>(B_ptr) % 32 == 0 && N % 8 == 0;
        const std::size_t row_blocks = (M + MR - 1) / MR;
        const std::size_t col_blocks = (N + NR - 1) / NR;
        const bool split_columns = row_blocks < pool.size() && col_blocks > 1;
        const std::size_t task_count = split_columns ? col_blocks : row_blocks;
        const std::size_t worker_count = std::min(pool.size(), task_count);

        auto worker = [&](std::size_t task_start, std::size_t task_end) {
            if (split_columns) {
                const std::size_t j_begin = task_start * NR;
                const std::size_t j_limit = std::min(task_end * NR, N);
                for (std::size_t i0 = 0; i0 < M; i0 += TI) {
                    const std::size_t i_end = std::min(i0 + TI, M);
                    for (std::size_t k0 = 0; k0 < K; k0 += TK) {
                        const std::size_t k_end = std::min(k0 + TK, K);
                        micro_kernels::mmap_matmul_tile(A_ptr, B_ptr, C_ptr, K, N,
                                         i0, i_end, j_begin, j_limit,
                                         k0, k_end, use_avx2, aligned_b);
                    }
                }
                return;
            }

            const std::size_t row_start = task_start * MR;
            const std::size_t row_end = std::min(task_end * MR, M);
            for (std::size_t i0 = row_start; i0 < row_end; i0 += TI) {
                const std::size_t i_end = std::min(i0 + TI, row_end);
                for (std::size_t j0 = 0; j0 < N; j0 += TJ) {
                    const std::size_t j_end = std::min(j0 + TJ, N);
                    for (std::size_t k0 = 0; k0 < K; k0 += TK) {
                        const std::size_t k_end = std::min(k0 + TK, K);
                        micro_kernels::mmap_matmul_tile(A_ptr, B_ptr, C_ptr, K, N,
                                         i0, i_end, j0, j_end,
                                         k0, k_end, use_avx2, aligned_b);
                    }
                }
            }
        };

        pool.run(worker_count, [&](std::size_t t) {
            const std::size_t task_start = task_count * t / worker_count;
            const std::size_t task_end = task_count * (t + 1) / worker_count;
            worker(task_start, task_end);
        });
        return result;
    }

    Mat transpose() const {
        Mat result(cols_, rows_);
        for (std::size_t r = 0; r < rows_; ++r) {
            for (std::size_t c = 0; c < cols_; ++c) {
                result.data_[c * rows_ + r] = data_[r * cols_ + c];
            }
        }
        return result;
    }

    // Extract columns [col_start, col_start + width) for all rows
    Mat slice_cols(std::size_t col_start, std::size_t width) const {
        if (col_start + width > cols_) {
            throw std::out_of_range("slice_cols: out of range");
        }
        Mat result(rows_, width);
        for (std::size_t r = 0; r < rows_; ++r) {
            for (std::size_t c = 0; c < width; ++c) {
                result.data_[r * width + c] = data_[r * cols_ + col_start + c];
            }
        }
        return result;
    }

    // Write src into columns [col_start, col_start + src.cols()) of this matrix
    void set_cols(std::size_t col_start, const Mat& src) {
        if (src.rows_ != rows_ || col_start + src.cols_ > cols_) {
            throw std::invalid_argument("set_cols: dimension mismatch");
        }
        for (std::size_t r = 0; r < rows_; ++r) {
            for (std::size_t c = 0; c < src.cols_; ++c) {
                data_[r * cols_ + col_start + c] = src.data_[r * src.cols_ + c];
            }
        }
    }

    // Element-wise addition: A + B -> C (same dimensions)
    Mat add(const Mat& B) const {
        if (rows_ != B.rows_ || cols_ != B.cols_) {
            throw std::invalid_argument("add: dimension mismatch");
        }
        Mat result(rows_, cols_);
        for (std::size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] + B.data_[i];
        }
        return result;
    }

    // Element-wise (Hadamard) product: A ⊙ B -> C (same dimensions)
    Mat hadamard(const Mat& B) const {
        if (rows_ != B.rows_ || cols_ != B.cols_) {
            throw std::invalid_argument("hadamard: dimension mismatch");
        }
        Mat result(rows_, cols_);
        for (std::size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] * B.data_[i];
        }
        return result;
    }

    // In-place SiLU(this) * B, avoiding two full-size temporary matrices.
    void silu_hadamard(const Mat& B, ThreadPool& pool) {
        if (rows_ != B.rows_ || cols_ != B.cols_) {
            throw std::invalid_argument("silu_hadamard: dimension mismatch");
        }
        const std::size_t n = data_.size();
        if (n == 0) {
            return;
        }
        auto worker = [&](std::size_t start, std::size_t end) {
            for (std::size_t i = start; i < end; ++i) {
                const T z = data_[i];
                data_[i] = (z / (T{1} + std::exp(-z))) * B.data_[i];
            }
        };

        const std::size_t worker_count = std::min(pool.size(), n);
        const std::size_t per_worker = (n + worker_count - 1) / worker_count;
        pool.run(worker_count, [&](std::size_t t) {
            const std::size_t start = t * per_worker;
            worker(start, std::min(start + per_worker, n));
        });
    }

    // Multiply each row by a (1, cols) row vector
    Mat mul_row_vec(const Mat& v) const {
        if (v.rows_ != 1 || v.cols_ != cols_) {
            throw std::invalid_argument("mul_row_vec: v must be (1, cols)");
        }
        Mat result(rows_, cols_);
        for (std::size_t r = 0; r < rows_; ++r) {
            for (std::size_t c = 0; c < cols_; ++c) {
                result.data_[r * cols_ + c] = data_[r * cols_ + c] * v.data_[c];
            }
        }
        return result;
    }

    // Apply a unary function element-wise, returning a new matrix
    template <typename Fn>
    Mat apply(Fn fn, ThreadPool& pool) const {
        Mat result(rows_, cols_);
        const std::size_t n = data_.size();
        if (n == 0) {
            return result;
        }

        auto worker = [&](std::size_t start, std::size_t end) {
            for (std::size_t i = start; i < end; ++i) {
                result.data_[i] = fn(data_[i]);
            }
        };

        const std::size_t worker_count = std::min(pool.size(), n);
        const std::size_t per_worker = (n + worker_count - 1) / worker_count;
        pool.run(worker_count, [&](std::size_t t) {
            const std::size_t start = t * per_worker;
            worker(start, std::min(start + per_worker, n));
        });

        return result;
    }

    // In-place scalar multiply
    void scale(T s) {
        for (auto& v : data_) v *= s;
    }

    // In-place: set values where mask is true to val
    void masked_fill(T val, bool upper_triangle) {
        for (std::size_t r = 0; r < rows_; ++r) {
            for (std::size_t c = 0; c < cols_; ++c) {
                if (upper_triangle && c > r) {
                    data_[r * cols_ + c] = val;
                }
            }
        }
    }

    // In-place softmax over a (n, 1) vector
    void softmax() {
        if (cols_ != 1) {
            throw std::invalid_argument("softmax: must be a column vector");
        }

        T max_val = *std::max_element(data_.begin(), data_.end());

        std::transform(data_.begin(), data_.end(), data_.begin(),
                       [max_val](T x) { return std::exp(x - max_val); });

        T sum = std::accumulate(data_.begin(), data_.end(), T{0});

        std::transform(data_.begin(), data_.end(), data_.begin(),
                       [sum](T x) { return x / sum; });
    }

    // In-place row-wise softmax over a (rows, cols) matrix
    void softmax_rows() {
        for (std::size_t r = 0; r < rows_; ++r) {
            auto row_begin = data_.begin() + r * cols_;
            auto row_end = row_begin + cols_;

            T max_val = *std::max_element(row_begin, row_end);
            std::transform(row_begin, row_end, row_begin,
                           [max_val](T x) { return std::exp(x - max_val); });
            T sum = std::accumulate(row_begin, row_end, T{0});
            std::transform(row_begin, row_end, row_begin,
                           [sum](T x) { return x / sum; });
        }
    }

private:
    std::size_t rows_;
    std::size_t cols_;
    std::vector<T> data_;
    T* mmap_data_ = nullptr; 
};
