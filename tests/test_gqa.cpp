#include "llama_transformer.h"
#include "llama_mmap_block.h"

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

// Brute-force reference: single-head attention (no GQA grouping logic).
// Q_h, K_h, V_h are (seq_len, head_dim).
Mat<float> ref_causal_attention(const Mat<float>& Q_h, const Mat<float>& K_h,
                                const Mat<float>& V_h, std::size_t head_dim) {
    std::size_t seq = Q_h.rows();
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // scores = Q_h @ K_h^T, scaled, causal-masked, softmaxed
    Mat<float> out(seq, head_dim);
    for (std::size_t r = 0; r < seq; ++r) {
        // compute row r of scores
        std::vector<float> row(seq);
        for (std::size_t c = 0; c < seq; ++c) {
            float dot = 0.0f;
            for (std::size_t k = 0; k < head_dim; ++k)
                dot += Q_h(r, k) * K_h(c, k);
            row[c] = (c > r) ? -std::numeric_limits<float>::infinity() : dot * scale;
        }
        // softmax
        float mx = *std::max_element(row.begin(), row.end());
        float sum = 0.0f;
        for (auto& v : row) { v = std::exp(v - mx); sum += v; }
        for (auto& v : row) v /= sum;
        // weighted sum of V rows
        for (std::size_t d = 0; d < head_dim; ++d) {
            float s = 0.0f;
            for (std::size_t c = 0; c < seq; ++c) s += row[c] * V_h(c, d);
            out(r, d) = s;
        }
    }
    return out;
}

} // namespace

// MHA (n_kv_heads == n_heads): compare GQA output against per-head reference.
TEST(GQATest, MHAMatchesReference) {
    std::mt19937 rng(42);

    constexpr std::size_t d_model = 8;
    constexpr std::size_t n_heads = 4;
    constexpr std::size_t n_kv_heads = 4; // standard MHA
    constexpr std::size_t head_dim = d_model / n_heads; // 2
    constexpr std::size_t seq_len = 5;

    auto W_q = random_mat(d_model, n_heads * head_dim, rng);
    auto W_k = random_mat(d_model, n_kv_heads * head_dim, rng);
    auto W_v = random_mat(d_model, n_kv_heads * head_dim, rng);
    auto W_o = random_mat(n_heads * head_dim, d_model, rng);
    auto x = random_mat(seq_len, d_model, rng);

    GQA<float> gqa(n_heads, n_kv_heads, head_dim,
                   W_q, W_k, W_v, W_o); // copy, keep originals

    Mat<float> result = gqa.forward(x);
    ASSERT_EQ(result.rows(), seq_len);
    ASSERT_EQ(result.cols(), d_model);

    // Build reference: project, apply RoPE, per-head attention, concat, output proj
    Mat<float> Q_all = x.matmul(W_q);
    Mat<float> K_all = x.matmul(W_k);
    Mat<float> V_all = x.matmul(W_v);

    Mat<float> concat(seq_len, n_heads * head_dim);
    for (std::size_t h = 0; h < n_heads; ++h) {
        auto Q_h = Q_all.slice_cols(h * head_dim, head_dim);
        auto K_h = K_all.slice_cols(h * head_dim, head_dim);
        auto V_h = V_all.slice_cols(h * head_dim, head_dim);
        apply_rope(Q_h);
        apply_rope(K_h);
        auto attn = ref_causal_attention(Q_h, K_h, V_h, head_dim);
        concat.set_cols(h * head_dim, attn);
    }
    Mat<float> expected = concat.matmul(W_o);

    for (std::size_t r = 0; r < seq_len; ++r)
        for (std::size_t c = 0; c < d_model; ++c)
            EXPECT_NEAR(result(r, c), expected(r, c), 1e-5f)
                << "mismatch at (" << r << ", " << c << ")";
}

// GQA (n_kv_heads < n_heads): verify KV head sharing produces correct result.
TEST(GQATest, GQAGroupedMatchesReference) {
    std::mt19937 rng(123);

    constexpr std::size_t d_model = 8;
    constexpr std::size_t n_heads = 4;
    constexpr std::size_t n_kv_heads = 2; // 2 KV heads shared across 4 query heads
    constexpr std::size_t head_dim = 2;
    constexpr std::size_t seq_len = 6;
    constexpr std::size_t n_groups = n_heads / n_kv_heads;

    auto W_q = random_mat(d_model, n_heads * head_dim, rng);
    auto W_k = random_mat(d_model, n_kv_heads * head_dim, rng);
    auto W_v = random_mat(d_model, n_kv_heads * head_dim, rng);
    auto W_o = random_mat(n_heads * head_dim, d_model, rng);
    auto x = random_mat(seq_len, d_model, rng);

    GQA<float> gqa(n_heads, n_kv_heads, head_dim, W_q, W_k, W_v, W_o);
    Mat<float> result = gqa.forward(x);

    // Reference with grouped KV sharing + RoPE
    Mat<float> Q_all = x.matmul(W_q);
    Mat<float> K_all = x.matmul(W_k);
    Mat<float> V_all = x.matmul(W_v);

    Mat<float> concat(seq_len, n_heads * head_dim);
    for (std::size_t h = 0; h < n_heads; ++h) {
        std::size_t kv_h = h / n_groups;
        auto Q_h = Q_all.slice_cols(h * head_dim, head_dim);
        auto K_h = K_all.slice_cols(kv_h * head_dim, head_dim);
        auto V_h = V_all.slice_cols(kv_h * head_dim, head_dim);
        apply_rope(Q_h);
        apply_rope(K_h);
        auto attn = ref_causal_attention(Q_h, K_h, V_h, head_dim);
        concat.set_cols(h * head_dim, attn);
    }
    Mat<float> expected = concat.matmul(W_o);

    for (std::size_t r = 0; r < seq_len; ++r)
        for (std::size_t c = 0; c < d_model; ++c)
            EXPECT_NEAR(result(r, c), expected(r, c), 1e-5f)
                << "mismatch at (" << r << ", " << c << ")";
}

// Output shape is correct.
TEST(GQATest, OutputShape) {
    std::mt19937 rng(7);

    constexpr std::size_t d_model = 16;
    constexpr std::size_t n_heads = 8;
    constexpr std::size_t n_kv_heads = 2;
    constexpr std::size_t head_dim = 2;
    constexpr std::size_t seq_len = 10;

    GQA<float> gqa(n_heads, n_kv_heads, head_dim,
                   random_mat(d_model, n_heads * head_dim, rng),
                   random_mat(d_model, n_kv_heads * head_dim, rng),
                   random_mat(d_model, n_kv_heads * head_dim, rng),
                   random_mat(n_heads * head_dim, d_model, rng));

    auto out = gqa.forward(random_mat(seq_len, d_model, rng));
    EXPECT_EQ(out.rows(), seq_len);
    EXPECT_EQ(out.cols(), d_model);
}

// RoPE: position 0 produces no rotation (cos=1, sin=0 for all pairs).
TEST(RoPETest, Position0IsIdentity) {
    // head_dim=4, seq_len=1 (single token at pos 0)
    Mat<float> x(1, 4, {1.0f, 2.0f, 3.0f, 4.0f});
    apply_rope(x); // pos_offset=0, so theta=0 for every pair
    EXPECT_NEAR(x(0, 0), 1.0f, 1e-6f);
    EXPECT_NEAR(x(0, 1), 2.0f, 1e-6f);
    EXPECT_NEAR(x(0, 2), 3.0f, 1e-6f);
    EXPECT_NEAR(x(0, 3), 4.0f, 1e-6f);
}

// RoPE: verify one rotation pair manually against the closed-form formula.
// For position p, pair j: theta = p / 10000^(2j/d).
// Rotated: (x0*cos - x1*sin, x0*sin + x1*cos).
TEST(RoPETest, SinglePairMatchesFormula) {
    constexpr float x0 = 1.5f, x1 = -0.5f;
    constexpr std::size_t pos = 3;
    constexpr std::size_t head_dim = 2; // one pair (j=0)
    // theta = pos / 10000^0 = 3.0
    float theta = static_cast<float>(pos);
    float cos_t = std::cos(theta);
    float sin_t = std::sin(theta);

    Mat<float> x(1, head_dim, {x0, x1});
    apply_rope(x, pos);

    EXPECT_NEAR(x(0, 0), x0 * cos_t - x1 * sin_t, 1e-5f);
    EXPECT_NEAR(x(0, 1), x0 * sin_t + x1 * cos_t, 1e-5f);
}

// RoPE: applying twice with pos_offset=0 then pos_offset=p should equal
// applying once with pos_offset=p on the original, since a second rotation
// by angle p gives cos(2p)/sin(2p) — NOT the expected single rotation.
// This test therefore verifies that apply_rope is NOT idempotent (a sanity check).
TEST(RoPETest, DoubleApplicationIsNotIdentity) {
    Mat<float> x(1, 2, {1.0f, 0.0f});
    Mat<float> x_copy(1, 2, {1.0f, 0.0f});
    apply_rope(x, 1); // rotate once by pos=1
    apply_rope(x, 1); // rotate again
    apply_rope(x_copy, 1); // single rotation
    // After two rotations the vector has rotated by 2*theta; it should differ.
    EXPECT_GT(std::abs(x(0, 0) - x_copy(0, 0)), 1e-4f);
}

// RoPE: pos_offset shifts all positions — row i is rotated as position (pos_offset+i).
TEST(RoPETest, PosOffsetShiftsPositions) {
    constexpr std::size_t seq_len = 3;
    constexpr std::size_t head_dim = 2;
    std::mt19937 rng(99);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> data(seq_len * head_dim);
    for (auto& v : data) v = dist(rng);

    Mat<float> x_offset(seq_len, head_dim, data);
    apply_rope(x_offset, /*pos_offset=*/5);

    // Build expected: apply per-row individually with the correct absolute position
    for (std::size_t i = 0; i < seq_len; ++i) {
        std::size_t pos = 5 + i;
        float theta = static_cast<float>(pos); // j=0, 10000^0 = 1
        float cos_t = std::cos(theta), sin_t = std::sin(theta);
        float orig0 = data[i * head_dim + 0];
        float orig1 = data[i * head_dim + 1];
        EXPECT_NEAR(x_offset(i, 0), orig0 * cos_t - orig1 * sin_t, 1e-5f)
            << "row " << i;
        EXPECT_NEAR(x_offset(i, 1), orig0 * sin_t + orig1 * cos_t, 1e-5f)
            << "row " << i;
    }
}

// RoPE: rotation preserves the norm of each token vector (it's an orthogonal transform).
TEST(RoPETest, PreservesNorm) {
    constexpr std::size_t seq_len = 4;
    constexpr std::size_t head_dim = 8;
    std::mt19937 rng(7);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> data(seq_len * head_dim);
    for (auto& v : data) v = dist(rng);

    Mat<float> x(seq_len, head_dim, data);

    // Compute norms before
    std::vector<float> norms_before(seq_len, 0.0f);
    for (std::size_t i = 0; i < seq_len; ++i)
        for (std::size_t d = 0; d < head_dim; ++d)
            norms_before[i] += data[i * head_dim + d] * data[i * head_dim + d];

    apply_rope(x);

    for (std::size_t i = 0; i < seq_len; ++i) {
        float norm_after = 0.0f;
        for (std::size_t d = 0; d < head_dim; ++d)
            norm_after += x(i, d) * x(i, d);
        EXPECT_NEAR(norm_after, norms_before[i], 1e-4f) << "norm changed at row " << i;
    }
}

TEST(RoPETest, MMapMultiHeadMatchesPerHeadReference) {
    constexpr std::size_t seq_len = 5;
    constexpr std::size_t n_heads = 3;
    constexpr std::size_t head_dim = 8;
    constexpr std::size_t pos_offset = 2;
    std::mt19937 rng(123);
    auto actual = random_mat(seq_len, n_heads * head_dim, rng);
    Mat<float> expected(seq_len, n_heads * head_dim);
    for (std::size_t r = 0; r < seq_len; ++r) {
        for (std::size_t c = 0; c < n_heads * head_dim; ++c) {
            expected(r, c) = actual(r, c);
        }
    }
    for (std::size_t h = 0; h < n_heads; ++h) {
        auto head = expected.slice_cols(h * head_dim, head_dim);
        gqa_mmap_detail::apply_rope(head, pos_offset);
        expected.set_cols(h * head_dim, head);
    }

    gqa_mmap_detail::apply_rope_heads(actual, n_heads, head_dim, pos_offset);
    for (std::size_t r = 0; r < seq_len; ++r) {
        for (std::size_t c = 0; c < n_heads * head_dim; ++c) {
            EXPECT_NEAR(actual(r, c), expected(r, c), 1e-6f)
                << "row=" << r << " col=" << c;
        }
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
