#include "mat.h"
#include "thread_pool.h"

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
    ThreadPool pool(3);
    gate.silu_hadamard(up, pool);
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

// Writes `values` (column-major: element (k, n) at values[n * K + k]) and maps it.
MMapWeight make_mapped_weight(const std::filesystem::path& directory,
                              std::size_t k, std::size_t n,
                              const std::vector<float>& values,
                              const std::string& layout = "column-major") {
    const std::string filename = "weight.bin";
    std::ofstream output(directory / filename, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(float)));
    output.close();

    nlohmann::json manifest = {
        {"tensors", nlohmann::json::array({{
            {"name", "weight"},
            {"dtype", "float32"},
            {"layout", layout},
            {"byte_order", "little-endian"},
            {"shape", {k, n}},
            {"size_bytes", values.size() * sizeof(float)},
            {"file", filename}
        }})}
    };
    MMapWeight weight;
    weight.load(directory, manifest, "weight");
    return weight;
}

// Independent scalar oracle: A(m, k) row-major times W(k, n) column-major, double accumulation.
std::vector<double> matmul_colmajor_ref(const std::vector<float>& a,
                                        const std::vector<float>& w,
                                        std::size_t m, std::size_t k, std::size_t n) {
    std::vector<double> c(m * n, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double acc = 0.0;
            for (std::size_t inner = 0; inner < k; ++inner) {
                acc += static_cast<double>(a[i * k + inner]) * w[j * k + inner];
            }
            c[i * n + j] = acc;
        }
    }
    return c;
}

std::vector<float> wave(std::size_t count, float freq, float scale, bool cosine) {
    std::vector<float> values(count);
    for (std::size_t i = 0; i < count; ++i) {
        const float x = static_cast<float>(i) * freq;
        values[i] = (cosine ? std::cos(x) : std::sin(x)) * scale;
    }
    return values;
}

void expect_matches_reference(std::size_t m, std::size_t k, std::size_t n,
                              const std::filesystem::path& directory,
                              std::initializer_list<std::size_t> thread_counts) {
    const auto a_values = wave(m * k, 0.17f, 0.2f, false);
    const auto w_values = wave(k * n, 0.11f, 0.3f, true);
    const Mat<float> a(m, k, a_values);
    const auto w = make_mapped_weight(directory, k, n, w_values);
    const auto expected = matmul_colmajor_ref(a_values, w_values, m, k, n);

    for (const std::size_t thread_count : thread_counts) {
        ThreadPool pool(thread_count);
        const auto actual = a.matmul(w, pool);
        ASSERT_EQ(actual.rows(), m);
        ASSERT_EQ(actual.cols(), n);
        for (std::size_t row = 0; row < m; ++row) {
            for (std::size_t col = 0; col < n; ++col) {
                const double e = expected[row * n + col];
                // fp32 accumulation error grows ~sqrt(k); scale the tolerance with it.
                const double tolerance = (2e-5 + 2e-5 * std::abs(e)) * std::sqrt(k / 64.0 + 1.0);
                ASSERT_NEAR(actual(row, col), e, tolerance)
                    << "shape=(" << m << ',' << k << ',' << n << ") threads="
                    << thread_count << " row=" << row << " col=" << col;
            }
        }
    }
}

} // namespace

TEST(MMapWeightTest, ColumnMajorAccessors) {
    TempMMapDirectory directory;
    // Logical W(3, 2) = [[1, 4], [2, 5], [3, 6]] stored column by column.
    const auto w = make_mapped_weight(directory.path(), 3, 2, {1, 2, 3, 4, 5, 6});
    EXPECT_EQ(w.in_features(), 3u);
    EXPECT_EQ(w.out_features(), 2u);
    EXPECT_EQ(w.col(0), w.data());
    EXPECT_EQ(w.col(1), w.data() + 3);
    EXPECT_FLOAT_EQ(w.col(1)[2], 6.0f);
}

TEST(MMapWeightTest, RejectsRowMajor) {
    TempMMapDirectory directory;
    EXPECT_THROW(make_mapped_weight(directory.path(), 3, 2, {1, 2, 3, 4, 5, 6}, "row-major"),
                 std::runtime_error);
}

TEST(MMapWeightTest, RejectsFormatVersion1) {
    TempMMapDirectory directory;
    const std::vector<float> values = {1, 2, 3, 4, 5, 6};
    std::ofstream(directory.path() / "t.bin", std::ios::binary)
        .write(reinterpret_cast<const char*>(values.data()), sizeof(float) * values.size());
    const nlohmann::json tensor = {
        {"name", "t"}, {"dtype", "float32"}, {"layout", "column-major"},
        {"byte_order", "little-endian"}, {"shape", {3, 2}},
        {"size_bytes", sizeof(float) * values.size()}, {"file", "t.bin"}};
    for (const int version : {1, kMMapFormatVersion}) {
        std::ofstream(directory.path() / "t.json")
            << nlohmann::json{{"format", "tinyserve-float32-matrices"},
                              {"format_version", version}, {"tensor", tensor}};
        if (version == kMMapFormatVersion) {
            EXPECT_NO_THROW(load_mapped_single_tensor(directory.path(), "t.json", "t"));
        } else {
            EXPECT_THROW(load_mapped_single_tensor(directory.path(), "t.json", "t"),
                         std::runtime_error);
        }
    }
}

TEST(MatTest, MMapMatmulKnownValues) {
    TempMMapDirectory directory;
    // A(2, 3) = [[1, 2, 3], [4, 5, 6]], W(3, 2) = [[1, 4], [2, 5], [3, 6]] (column-major bytes).
    const Mat<float> a(2, 3, {1, 2, 3, 4, 5, 6});
    const auto w = make_mapped_weight(directory.path(), 3, 2, {1, 2, 3, 4, 5, 6});
    ThreadPool pool(2);
    const auto c = a.matmul(w, pool);
    EXPECT_FLOAT_EQ(c(0, 0), 14.0f);
    EXPECT_FLOAT_EQ(c(0, 1), 32.0f);
    EXPECT_FLOAT_EQ(c(1, 0), 32.0f);
    EXPECT_FLOAT_EQ(c(1, 1), 77.0f);
}

TEST(MatTest, MMapMatmulMatchesReferenceSmallShapes) {
    TempMMapDirectory directory;
    const std::vector<std::array<std::size_t, 3>> shapes = {
        {1, 1, 1}, {1, 7, 15}, {1, 15, 16}, {1, 16, 17}, {1, 17, 20},
        {1, 64, 16}, {1, 65, 17}, {1, 130, 70},
        {5, 7, 15}, {6, 64, 16}, {7, 65, 17}, {65, 130, 70}
    };
    for (const auto [m, k, n] : shapes) {
        expect_matches_reference(m, k, n, directory.path(), {1, 3, 30});
    }
}

// SmolLM3 inner dimensions (d_model = 2048, d_ff = 11008) for decode (M = 1) and prefill,
// with N deliberately not a multiple of any plausible micro-tile width.
TEST(MatTest, MMapMatmulMatchesReferenceModelShapes) {
    TempMMapDirectory directory;
    const std::vector<std::array<std::size_t, 3>> shapes = {
        {1, 2048, 37}, {7, 2048, 37}, {64, 2048, 37},
        {1, 11008, 13}, {7, 11008, 13}, {64, 11008, 13},
        // K straddles the 512 K-tile with a non-multiple-of-8 tail; odd M and N > one J tile.
        {9, 1031, 70}, {1, 1031, 70}
    };
    for (const auto [m, k, n] : shapes) {
        expect_matches_reference(m, k, n, directory.path(), {1, 4});
    }
}

TEST(MatTest, MMapMatmulPromotesFloatWeightsForDoubleInput) {
    TempMMapDirectory directory;
    const Mat<double> a(2, 3, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    const auto w = make_mapped_weight(directory.path(), 3, 2, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    ThreadPool pool(4);
    const auto c = a.matmul(w, pool);
    EXPECT_DOUBLE_EQ(c(0, 0), 14.0);
    EXPECT_DOUBLE_EQ(c(0, 1), 32.0);
    EXPECT_DOUBLE_EQ(c(1, 0), 32.0);
    EXPECT_DOUBLE_EQ(c(1, 1), 77.0);
}

TEST(MatTest, MMapMatmulValidatesShape) {
    TempMMapDirectory directory;
    const auto w = make_mapped_weight(directory.path(), 3, 2, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    const Mat<float> wrong_inner_dimension(1, 2);
    ThreadPool pool(2);
    EXPECT_THROW(wrong_inner_dimension.matmul(w, pool), std::invalid_argument);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
