#include "mat.h"

#include <gtest/gtest.h>

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

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
