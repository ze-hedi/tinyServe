#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <vector>

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

    // Micro-kernel: C[i0..i_end, j0..j_end] += A[.., k0..k_end] * B[k0..k_end, ..]
    __attribute__((always_inline)) static inline void matmul_tile(const T* A, const T* B, T* C, std::size_t K, std::size_t N,
                            std::size_t i0, std::size_t i_end, std::size_t j0,
                            std::size_t j_end, std::size_t k0, std::size_t k_end) {
        for (std::size_t i = i0; i < i_end; ++i) {
            for (std::size_t k = k0; k < k_end; ++k) {
                const T a = A[i * K + k];
                const std::size_t b_row = k * N;
                const std::size_t c_row = i * N;
                for (std::size_t j = j0; j < j_end; ++j) {
                    C[c_row + j] += a * B[b_row + j];
                }
            }
        }
    }

    // A(m, k) * B(k, n) -> C(m, n)  — 3D tiled for cache locality
    Mat matmul(const Mat& B) const {
        if (cols_ != B.rows_) {
            throw std::invalid_argument("matmul: dimension mismatch");
        }

        const std::size_t M = rows_;
        const std::size_t K = cols_;
        const std::size_t N = B.cols_;
        constexpr std::size_t TI = 64;
        constexpr std::size_t TJ = 64;
        constexpr std::size_t TK = 64;

        constexpr std::size_t NUM_THREADS = 30;

        Mat result(M, N);
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

        const std::size_t rows_per_thread = (M + NUM_THREADS - 1) / NUM_THREADS;
        std::vector<std::thread> threads;
        threads.reserve(NUM_THREADS);
        for (std::size_t t = 0; t < NUM_THREADS; ++t) {
            const std::size_t row_start = t * rows_per_thread;
            const std::size_t row_end = std::min(row_start + rows_per_thread, M);
            if (row_start >= M) break;
            threads.emplace_back(worker, row_start, row_end);
        }
        for (auto& th : threads) {
            th.join();
        }
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
    Mat apply(Fn fn) const {
        Mat result(rows_, cols_);
        for (std::size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = fn(data_[i]);
        }
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
};
