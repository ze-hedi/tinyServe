#include "mat.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

TEST(MatTest, ZeroInit) {
    Mat<float> m(2, 3);
    EXPECT_EQ(m.rows(), 2);
    EXPECT_EQ(m.cols(), 3);
    for (std::size_t r = 0; r < 2; ++r)
        for (std::size_t c = 0; c < 3; ++c)
            EXPECT_FLOAT_EQ(m(r, c), 0.0f);
}

TEST(MatTest, FromData) {
    Mat<float> m(2, 3, {1, 2, 3, 4, 5, 6});
    EXPECT_FLOAT_EQ(m(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(m(0, 2), 3.0f);
    EXPECT_FLOAT_EQ(m(1, 1), 5.0f);
}

TEST(MatTest, Matvec) {
    // [1 2 3]   [1]   [14]
    // [4 5 6] * [2] = [32]
    //           [3]
    Mat<float> A(2, 3, {1, 2, 3, 4, 5, 6});
    Mat<float> v(3, 1, {1, 2, 3});
    auto result = A.matvec(v);

    EXPECT_EQ(result.rows(), 2);
    EXPECT_EQ(result.cols(), 1);
    EXPECT_FLOAT_EQ(result(0, 0), 14.0f);
    EXPECT_FLOAT_EQ(result(1, 0), 32.0f);
}

TEST(MatTest, MatvecDimMismatch) {
    Mat<float> A(2, 3, {1, 2, 3, 4, 5, 6});
    Mat<float> v(2, 1, {1, 2});
    EXPECT_THROW(A.matvec(v), std::invalid_argument);
}

TEST(MatTest, MatvecNotVector) {
    Mat<float> A(2, 3, {1, 2, 3, 4, 5, 6});
    Mat<float> B(3, 2, {1, 2, 3, 4, 5, 6});
    EXPECT_THROW(A.matvec(B), std::invalid_argument);
}

TEST(MatTest, Softmax) {
    // softmax([1, 2, 3]) = [e^1, e^2, e^3] / sum
    Mat<float> v(3, 1, {1.0f, 2.0f, 3.0f});
    v.softmax();

    // Values should sum to 1
    float sum = v(0, 0) + v(1, 0) + v(2, 0);
    EXPECT_NEAR(sum, 1.0f, 1e-6f);

    // Monotonicity preserved
    EXPECT_LT(v(0, 0), v(1, 0));
    EXPECT_LT(v(1, 0), v(2, 0));

    // Check known values: softmax([1,2,3])
    float e1 = std::exp(1.0f), e2 = std::exp(2.0f), e3 = std::exp(3.0f);
    float s = e1 + e2 + e3;
    EXPECT_NEAR(v(0, 0), e1 / s, 1e-6f);
    EXPECT_NEAR(v(1, 0), e2 / s, 1e-6f);
    EXPECT_NEAR(v(2, 0), e3 / s, 1e-6f);
}

TEST(MatTest, SoftmaxLargeValues) {
    // Numerical stability: large values shouldn't overflow
    Mat<float> v(3, 1, {1000.0f, 1001.0f, 1002.0f});
    v.softmax();

    float sum = v(0, 0) + v(1, 0) + v(2, 0);
    EXPECT_NEAR(sum, 1.0f, 1e-6f);
}

TEST(MatTest, SoftmaxNotVector) {
    Mat<float> m(2, 3, {1, 2, 3, 4, 5, 6});
    EXPECT_THROW(m.softmax(), std::invalid_argument);
}

TEST(MatTest, SiluHadamardInPlace) {
    Mat<float> gate(2, 3, {-2.0f, -0.5f, 0.0f, 0.5f, 2.0f, 20.0f});
    const Mat<float> up(2, 3, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    const std::vector<float> original = {-2.0f, -0.5f, 0.0f, 0.5f, 2.0f, 20.0f};
    std::vector<std::thread> threads;
    gate.silu_hadamard(up, &threads, 3);
    for (std::size_t r = 0; r < gate.rows(); ++r) {
        for (std::size_t c = 0; c < gate.cols(); ++c) {
            const std::size_t i = r * gate.cols() + c;
            const float z = original[i];
            const float expected = (z / (1.0f + std::exp(-z))) * up(r, c);
            EXPECT_NEAR(gate(r, c), expected, 1e-6f);
        }
    }
}

namespace {

class TempMMapDirectory {
public:
    TempMMapDirectory() {
        for (std::size_t suffix = 0; suffix < 100; ++suffix) {
            path_ = std::filesystem::current_path() /
                    ("tinyserve_mmap_mat_test_" + std::to_string(::getpid()) + "_" +
                     std::to_string(suffix));
            if (std::filesystem::create_directory(path_)) {
                return;
            }
        }
        throw std::runtime_error("cannot create a unique mmap test directory");
    }

    ~TempMMapDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }

    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

MMapWeight make_mapped_weight(const std::filesystem::path& directory,
                              std::size_t rows, std::size_t cols,
                              const std::vector<float>& values) {
    const std::string filename = "weight.bin";
    std::ofstream output(directory / filename, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(float)));
    output.close();

    nlohmann::json manifest = {
        {"tensors", nlohmann::json::array({{
            {"name", "weight"},
            {"dtype", "float32"},
            {"layout", "row-major"},
            {"byte_order", "little-endian"},
            {"shape", {rows, cols}},
            {"size_bytes", values.size() * sizeof(float)},
            {"file", filename}
        }})}
    };
    MMapWeight weight;
    weight.load(directory, manifest, "weight");
    return weight;
}

} // namespace

TEST(MatTest, MMapMatmulMatchesIndependentReference) {
    TempMMapDirectory directory;
    const std::vector<std::array<std::size_t, 3>> shapes = {
        {1, 1, 1}, {5, 7, 15}, {6, 64, 16}, {7, 65, 17}, {65, 130, 70}
    };

    for (const auto [m, k, n] : shapes) {
        std::vector<float> a_values(m * k);
        std::vector<float> b_values(k * n);
        for (std::size_t i = 0; i < a_values.size(); ++i) {
            a_values[i] = std::sin(static_cast<float>(i) * 0.17f) * 0.2f;
        }
        for (std::size_t i = 0; i < b_values.size(); ++i) {
            b_values[i] = std::cos(static_cast<float>(i) * 0.11f) * 0.3f;
        }
        const Mat<float> a(m, k, a_values);
        auto b = make_mapped_weight(directory.path(), k, n, b_values);

        for (const std::size_t thread_count : {std::size_t{1}, std::size_t{3}, std::size_t{30}}) {
            const auto actual = a.matmul(b, nullptr, thread_count);
            for (std::size_t row = 0; row < m; ++row) {
                for (std::size_t col = 0; col < n; ++col) {
                    double expected = 0.0;
                    for (std::size_t inner = 0; inner < k; ++inner) {
                        expected += static_cast<double>(a(row, inner)) * b_values[inner * n + col];
                    }
                    const double tolerance = 2e-5 + 2e-5 * std::abs(expected);
                    ASSERT_NEAR(actual(row, col), expected, tolerance)
                        << "shape=(" << m << ',' << k << ',' << n << ") threads="
                        << thread_count << " row=" << row << " col=" << col;
                }
            }
        }
    }
}

TEST(MatTest, MMapMatmulPromotesFloatWeightsForDoubleInput) {
    TempMMapDirectory directory;
    const Mat<double> a(2, 3, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    auto b = make_mapped_weight(directory.path(), 3, 2, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    const auto result = a.matmul(b, nullptr, 4);
    EXPECT_DOUBLE_EQ(result(0, 0), 22.0);
    EXPECT_DOUBLE_EQ(result(0, 1), 28.0);
    EXPECT_DOUBLE_EQ(result(1, 0), 49.0);
    EXPECT_DOUBLE_EQ(result(1, 1), 64.0);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
