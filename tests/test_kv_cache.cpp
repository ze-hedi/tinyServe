#include "kv_cache.h"
#include "llama_mmap_block.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

class TempDir {
public:
    TempDir() {
        path_ = std::filesystem::temp_directory_path() /
                ("tinyserve_kv_" + std::to_string(::getpid()));
        std::filesystem::remove_all(path_);
        std::filesystem::create_directories(path_);
    }
    ~TempDir() { std::filesystem::remove_all(path_); }
    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

MMapWeight make_weight(const std::filesystem::path& dir, const std::string& name,
                       std::size_t rows, std::size_t cols, std::mt19937& rng, float stddev) {
    std::normal_distribution<float> dist(0.0f, stddev);
    std::vector<float> values(rows * cols);
    for (auto& v : values) v = dist(rng);
    const std::string file = name + ".bin";
    std::ofstream out(dir / file, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(float)));
    out.close();
    nlohmann::json manifest = {{"tensors", nlohmann::json::array({{
        {"name", name}, {"dtype", "float32"}, {"layout", "row-major"},
        {"byte_order", "little-endian"}, {"shape", {rows, cols}},
        {"size_bytes", values.size() * sizeof(float)}, {"file", file}}})}};
    MMapWeight weight;
    weight.load(dir, manifest, name);
    return weight;
}

Mat<float> random_mat(std::size_t rows, std::size_t cols, std::mt19937& rng) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> data(rows * cols);
    for (auto& v : data) v = dist(rng);
    return Mat<float>(rows, cols, std::move(data));
}

Mat<float> rows_of(const Mat<float>& m, std::size_t begin, std::size_t end) {
    Mat<float> out(end - begin, m.cols());
    for (std::size_t r = begin; r < end; ++r)
        for (std::size_t c = 0; c < m.cols(); ++c) out(r - begin, c) = m(r, c);
    return out;
}

constexpr std::size_t kModel = 16, kHeads = 4, kKvHeads = 2, kHeadDim = 4, kFf = 32;

// Runs `x` through the block in chunks of the given sizes (prefill, then decode steps) and
// compares every output row with the single uncached pass over the whole sequence.
void expect_cached_matches_full(bool use_rope, const std::vector<std::size_t>& chunks) {
    TempDir dir;
    std::mt19937 rng(7);
    MMapLlamaBlockWeights w;
    w.input_layernorm = make_weight(dir.path(), "in_ln", 1, kModel, rng, 0.3f);
    w.q_proj = make_weight(dir.path(), "q", kModel, kHeads * kHeadDim, rng, 0.4f);
    w.k_proj = make_weight(dir.path(), "k", kModel, kKvHeads * kHeadDim, rng, 0.4f);
    w.v_proj = make_weight(dir.path(), "v", kModel, kKvHeads * kHeadDim, rng, 0.4f);
    w.o_proj = make_weight(dir.path(), "o", kHeads * kHeadDim, kModel, rng, 0.4f);
    w.post_attention_layernorm = make_weight(dir.path(), "post_ln", 1, kModel, rng, 0.3f);
    w.gate_proj = make_weight(dir.path(), "gate", kModel, kFf, rng, 0.3f);
    w.up_proj = make_weight(dir.path(), "up", kModel, kFf, rng, 0.3f);
    w.down_proj = make_weight(dir.path(), "down", kFf, kModel, rng, 0.3f);

    ThreadPool pool(3);
    LlamaBlock_MMap<float> block(kHeads, kKvHeads, kHeadDim, w, pool, 10000.0f, use_rope);

    std::size_t total = 0;
    for (auto n : chunks) total += n;
    const Mat<float> x = random_mat(total, kModel, rng);
    const Mat<float> expected = block.forward(x);

    KVCache<float> cache(/*n_layers=*/1, kKvHeads * kHeadDim, total);
    std::size_t at = 0;
    for (auto n : chunks) {
        const Mat<float> out = block.forward(rows_of(x, at, at + n), &cache, 0);
        cache.advance(n);
        for (std::size_t r = 0; r < n; ++r)
            for (std::size_t c = 0; c < kModel; ++c)
                EXPECT_NEAR(out(r, c), expected(at + r, c), 1e-4f)
                    << "row " << at + r << " col " << c << " (chunk of " << n << ")";
        at += n;
    }
    EXPECT_EQ(cache.length(), total);
}

}  // namespace

TEST(KVCacheTest, PrefillThenSingleTokenDecodeMatchesFullPass) {
    expect_cached_matches_full(true, {5, 1, 1, 1});
}

TEST(KVCacheTest, DecodeWithoutRopeMatchesFullPass) {  // NoPE layers
    expect_cached_matches_full(false, {5, 1, 1, 1});
}

TEST(KVCacheTest, UnevenChunksMatchFullPass) {  // cached path with several new rows at once
    expect_cached_matches_full(true, {3, 4, 1, 2});
}

TEST(KVCacheTest, TokenByTokenFromEmptyMatchesFullPass) {
    expect_cached_matches_full(true, {1, 1, 1, 1, 1, 1});
}

TEST(KVCacheTest, AdvanceBeyondCapacityThrows) {
    KVCache<float> cache(2, 8, 4);
    cache.advance(4);
    EXPECT_THROW(cache.advance(1), std::length_error);
    cache.reset();
    EXPECT_EQ(cache.length(), 0u);
}
