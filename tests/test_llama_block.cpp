#include "llama_transformer.h"
#include "thread_pool.h"

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include <gtest/gtest.h>

namespace {

Mat<float> random_mat(std::size_t rows, std::size_t cols, std::mt19937& rng) {
    std::normal_distribution<float> dist(0.0f, 0.5f);
    std::vector<float> data(rows * cols);
    for (auto& v : data) v = dist(rng);
    return Mat<float>(rows, cols, std::move(data));
}

Mat<float> ones_row(std::size_t cols) {
    std::vector<float> data(cols, 1.0f);
    return Mat<float>(1, cols, std::move(data));
}

} // namespace

// --- Mat primitive tests ---

TEST(MatOpsTest, AddElementwise) {
    Mat<float> a(2, 3, {1, 2, 3, 4, 5, 6});
    Mat<float> b(2, 3, {10, 20, 30, 40, 50, 60});
    auto c = a.add(b);
    EXPECT_FLOAT_EQ(c(0, 0), 11);
    EXPECT_FLOAT_EQ(c(1, 2), 66);
}

TEST(MatOpsTest, HadamardProduct) {
    Mat<float> a(2, 2, {1, 2, 3, 4});
    Mat<float> b(2, 2, {5, 6, 7, 8});
    auto c = a.hadamard(b);
    EXPECT_FLOAT_EQ(c(0, 0), 5);
    EXPECT_FLOAT_EQ(c(0, 1), 12);
    EXPECT_FLOAT_EQ(c(1, 0), 21);
    EXPECT_FLOAT_EQ(c(1, 1), 32);
}

TEST(MatOpsTest, MulRowVec) {
    Mat<float> a(3, 2, {1, 2, 3, 4, 5, 6});
    Mat<float> v(1, 2, {10, 100});
    auto c = a.mul_row_vec(v);
    EXPECT_FLOAT_EQ(c(0, 0), 10);
    EXPECT_FLOAT_EQ(c(0, 1), 200);
    EXPECT_FLOAT_EQ(c(2, 0), 50);
    EXPECT_FLOAT_EQ(c(2, 1), 600);
}

TEST(MatOpsTest, Apply) {
    Mat<float> a(1, 3, {-1, 0, 2});
    ThreadPool pool(2);
    auto b = a.apply([](float x) { return x * x; }, pool);
    EXPECT_FLOAT_EQ(b(0, 0), 1);
    EXPECT_FLOAT_EQ(b(0, 1), 0);
    EXPECT_FLOAT_EQ(b(0, 2), 4);
}

// --- RMSNorm tests ---

TEST(RMSNormTest, GammaOnesMatchesManual) {
    // With gamma=1, RMSNorm(x) = x / rms(x)
    std::size_t d = 4;
    Mat<float> x(1, d, {1.0f, 2.0f, 3.0f, 4.0f});
    RMSNorm<float> norm(ones_row(d));

    ThreadPool pool(2);
    auto out = norm.forward(x, pool);

    // rms = sqrt(mean(1+4+9+16) + 1e-6) = sqrt(7.5 + 1e-6)
    float rms = std::sqrt(30.0f / 4.0f + 1e-6f);
    for (std::size_t c = 0; c < d; ++c) {
        EXPECT_NEAR(out(0, c), x(0, c) / rms, 1e-5f);
    }
}

TEST(RMSNormTest, GammaScales) {
    std::size_t d = 2;
    Mat<float> x(1, d, {3.0f, 4.0f});
    Mat<float> gamma(1, d, {2.0f, 0.5f});
    RMSNorm<float> norm(std::move(gamma));

    ThreadPool pool(2);
    auto out = norm.forward(x, pool);
    float rms = std::sqrt((9.0f + 16.0f) / 2.0f + 1e-6f);
    EXPECT_NEAR(out(0, 0), (3.0f / rms) * 2.0f, 1e-5f);
    EXPECT_NEAR(out(0, 1), (4.0f / rms) * 0.5f, 1e-5f);
}

TEST(RMSNormTest, OutputShape) {
    std::mt19937 rng(1);
    std::size_t d = 8;
    RMSNorm<float> norm(ones_row(d));
    auto x = random_mat(5, d, rng);
    ThreadPool pool(2);
    auto out = norm.forward(x, pool);
    EXPECT_EQ(out.rows(), 5u);
    EXPECT_EQ(out.cols(), d);
}

// --- SwiGLU tests ---

TEST(SwiGLUTest, SiluActivation) {
    // Verify silu(0) = 0, silu is odd-ish, and silu(large) ≈ large
    float z = 0.0f;
    float silu_z = z / (1.0f + std::exp(-z));
    EXPECT_NEAR(silu_z, 0.0f, 1e-6f);

    z = 5.0f;
    silu_z = z / (1.0f + std::exp(-z));
    EXPECT_NEAR(silu_z, z, 0.05f); // sigmoid(5) ≈ 0.993
}

TEST(SwiGLUTest, OutputShape) {
    std::mt19937 rng(42);
    std::size_t d_model = 8, d_ff = 16, seq_len = 5;

    SwiGLU<float> mlp(random_mat(d_model, d_ff, rng),
                      random_mat(d_model, d_ff, rng),
                      random_mat(d_ff, d_model, rng));

    auto x = random_mat(seq_len, d_model, rng);
    ThreadPool pool(4);
    auto out = mlp.forward(x, pool);
    EXPECT_EQ(out.rows(), seq_len);
    EXPECT_EQ(out.cols(), d_model);
}

TEST(SwiGLUTest, MatchesReference) {
    std::mt19937 rng(7);
    std::size_t d_model = 4, d_ff = 8, seq_len = 3;

    auto W_gate = random_mat(d_model, d_ff, rng);
    auto W_up = random_mat(d_model, d_ff, rng);
    auto W_down = random_mat(d_ff, d_model, rng);
    auto x = random_mat(seq_len, d_model, rng);

    SwiGLU<float> mlp(W_gate, W_up, W_down);
    ThreadPool pool(4);
    auto result = mlp.forward(x, pool);

    // Reference computation
    auto gate = x.matmul(W_gate, pool);
    auto up = x.matmul(W_up, pool);
    auto activated = gate.apply([](float z) { return z / (1.0f + std::exp(-z)); }, pool);
    auto fused = activated.hadamard(up);
    auto expected = fused.matmul(W_down, pool);

    for (std::size_t r = 0; r < seq_len; ++r)
        for (std::size_t c = 0; c < d_model; ++c)
            EXPECT_NEAR(result(r, c), expected(r, c), 1e-5f);
}

// --- LlamaBlock tests ---

TEST(LlamaBlockTest, OutputShape) {
    std::mt19937 rng(99);
    constexpr std::size_t d_model = 8, n_heads = 4, n_kv_heads = 2;
    constexpr std::size_t head_dim = d_model / n_heads; // 2
    constexpr std::size_t d_ff = 16, seq_len = 5;

    RMSNorm<float> attn_norm(ones_row(d_model));
    GQA<float> attn(n_heads, n_kv_heads, head_dim,
                    random_mat(d_model, n_heads * head_dim, rng),
                    random_mat(d_model, n_kv_heads * head_dim, rng),
                    random_mat(d_model, n_kv_heads * head_dim, rng),
                    random_mat(n_heads * head_dim, d_model, rng));
    RMSNorm<float> ffn_norm(ones_row(d_model));
    SwiGLU<float> mlp(random_mat(d_model, d_ff, rng),
                      random_mat(d_model, d_ff, rng),
                      random_mat(d_ff, d_model, rng));

    ThreadPool pool(4);
    LlamaBlock<float> block(std::move(attn_norm), std::move(attn),
                            std::move(ffn_norm), std::move(mlp), pool);

    auto x = random_mat(seq_len, d_model, rng);
    auto out = block.forward(x);
    EXPECT_EQ(out.rows(), seq_len);
    EXPECT_EQ(out.cols(), d_model);
}

TEST(LlamaBlockTest, ResidualConnection) {
    // With zero weights, GQA and MLP produce zeros -> output == input (residual only)
    constexpr std::size_t d_model = 4, n_heads = 2, n_kv_heads = 2;
    constexpr std::size_t head_dim = 2, d_ff = 8, seq_len = 3;

    auto zero = [](std::size_t r, std::size_t c) {
        return Mat<float>(r, c, std::vector<float>(r * c, 0.0f));
    };

    RMSNorm<float> attn_norm(ones_row(d_model));
    GQA<float> attn(n_heads, n_kv_heads, head_dim,
                    zero(d_model, n_heads * head_dim),
                    zero(d_model, n_kv_heads * head_dim),
                    zero(d_model, n_kv_heads * head_dim),
                    zero(n_heads * head_dim, d_model));
    RMSNorm<float> ffn_norm(ones_row(d_model));
    SwiGLU<float> mlp(zero(d_model, d_ff), zero(d_model, d_ff), zero(d_ff, d_model));

    ThreadPool pool(4);
    LlamaBlock<float> block(std::move(attn_norm), std::move(attn),
                            std::move(ffn_norm), std::move(mlp), pool);

    Mat<float> x(seq_len, d_model, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto out = block.forward(x);

    for (std::size_t r = 0; r < seq_len; ++r)
        for (std::size_t c = 0; c < d_model; ++c)
            EXPECT_FLOAT_EQ(out(r, c), x(r, c))
                << "residual broken at (" << r << ", " << c << ")";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
