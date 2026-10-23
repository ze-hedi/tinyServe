#include "embeddings.h"

#include <gtest/gtest.h>

// Small vocabulary for testing: 4 tokens, embed_dim 3
// Token 0: [1, 2, 3]
// Token 1: [4, 5, 6]
// Token 2: [7, 8, 9]
// Token 3: [10, 11, 12]
static Embedding<float> make_test_embedding() {
    return Embedding<float>(4, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
}

TEST(EmbeddingTest, Dimensions) {
    auto emb = make_test_embedding();
    EXPECT_EQ(emb.vocab_size(), 4);
    EXPECT_EQ(emb.embed_dim(), 3);
}

TEST(EmbeddingTest, SingleToken) {
    auto emb = make_test_embedding();
    auto out = emb.forward({2});

    EXPECT_EQ(out.rows(), 1);
    EXPECT_EQ(out.cols(), 3);
    EXPECT_FLOAT_EQ(out(0, 0), 7.0f);
    EXPECT_FLOAT_EQ(out(0, 1), 8.0f);
    EXPECT_FLOAT_EQ(out(0, 2), 9.0f);
}

TEST(EmbeddingTest, Sequence) {
    auto emb = make_test_embedding();
    auto out = emb.forward({0, 3, 1});

    EXPECT_EQ(out.rows(), 3);
    EXPECT_EQ(out.cols(), 3);

    // Row 0 = token 0: [1, 2, 3]
    EXPECT_FLOAT_EQ(out(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(out(0, 1), 2.0f);
    EXPECT_FLOAT_EQ(out(0, 2), 3.0f);

    // Row 1 = token 3: [10, 11, 12]
    EXPECT_FLOAT_EQ(out(1, 0), 10.0f);
    EXPECT_FLOAT_EQ(out(1, 1), 11.0f);
    EXPECT_FLOAT_EQ(out(1, 2), 12.0f);

    // Row 2 = token 1: [4, 5, 6]
    EXPECT_FLOAT_EQ(out(2, 0), 4.0f);
    EXPECT_FLOAT_EQ(out(2, 1), 5.0f);
    EXPECT_FLOAT_EQ(out(2, 2), 6.0f);
}

TEST(EmbeddingTest, DuplicateTokens) {
    auto emb = make_test_embedding();
    auto out = emb.forward({1, 1});

    EXPECT_EQ(out.rows(), 2);
    for (std::size_t j = 0; j < 3; ++j) {
        EXPECT_FLOAT_EQ(out(0, j), out(1, j));
    }
}

TEST(EmbeddingTest, OutOfRange) {
    auto emb = make_test_embedding();
    EXPECT_THROW(emb.forward({4}), std::out_of_range);
    EXPECT_THROW(emb.forward({0, 100}), std::out_of_range);
}

TEST(EmbeddingTest, EmptySequence) {
    auto emb = make_test_embedding();
    EXPECT_THROW(emb.forward({}), std::invalid_argument);
}

TEST(EmbeddingTest, WeightDimMismatch) {
    // 4 tokens * 3 dims = 12 elements, but only 6 given
    EXPECT_THROW(Embedding<float>(4, 3, {1, 2, 3, 4, 5, 6}), std::invalid_argument);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
