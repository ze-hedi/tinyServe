// #include "tokenizer.h"

// #include <gtest/gtest.h>
// #include <string>
// #include <vector>

// TEST(Tokenizer, PreTokenizeBasic) {
//     Tokenizer tok;
//     auto chunks = tok.pre_tokenize("Hello world");
//     ASSERT_EQ(chunks.size(), 2);
//     EXPECT_EQ(chunks[0], "Hello");
//     EXPECT_EQ(chunks[1], " world");
// }

// TEST(Tokenizer, PreTokenizeContractions) {
//     Tokenizer tok;
//     auto chunks = tok.pre_tokenize("I'm don't he's");
//     // "I" "'m" " don" "'t" " he" "'s"
//     ASSERT_EQ(chunks.size(), 6);
//     EXPECT_EQ(chunks[0], "I");
//     EXPECT_EQ(chunks[1], "'m");
//     EXPECT_EQ(chunks[2], " don");
//     EXPECT_EQ(chunks[3], "'t");
//     EXPECT_EQ(chunks[4], " he");
//     EXPECT_EQ(chunks[5], "'s");
// }

// TEST(Tokenizer, PreTokenizeNumbers) {
//     Tokenizer tok;
//     auto chunks = tok.pre_tokenize("test 12345 ok");
//     // numbers split into groups of up to 3 digits
//     // "test" " " "123" "45" " ok"
//     ASSERT_GE(chunks.size(), 4);
//     EXPECT_EQ(chunks[0], "test");
// }

// TEST(Tokenizer, PreTokenizeWhitespace) {
//     Tokenizer tok;
//     auto chunks = tok.pre_tokenize("Hello   world  ");
//     // With lookahead: "Hello" "  " " world" "  "
//     ASSERT_EQ(chunks.size(), 4);
//     EXPECT_EQ(chunks[0], "Hello");
//     EXPECT_EQ(chunks[1], "  ");
//     EXPECT_EQ(chunks[2], " world");
//     EXPECT_EQ(chunks[3], "  ");
// }

// TEST(Tokenizer, PreTokenizePunctuation) {
//     Tokenizer tok;
//     auto chunks = tok.pre_tokenize("hello, world!");
//     ASSERT_GE(chunks.size(), 3);
//     EXPECT_EQ(chunks[0], "hello");
// }

// int main(int argc, char** argv) {
//     testing::InitGoogleTest(&argc, argv);
//     return RUN_ALL_TESTS();
// }
