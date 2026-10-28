#pragma once

#include "cuda_utils.h"
#include "gpu_kernels.h"
#include "gpu_kv_cache.h"
#include "mmap_block_loader.h"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpu {

struct GpuModelConfig {
    std::size_t n_heads = 16;
    std::size_t n_kv_heads = 4;
    std::size_t head_dim = 128;
    std::size_t d_model = 2048;  // n_heads * head_dim
    float rope_theta = 5000000.0f;
    float rms_eps = 1e-6f;
};

// One transformer block to run on the GPU: its position in the full model (SmolLM3 skips RoPE
// on every 4th layer, so the global index matters) and its mmap weights.
struct GpuLayerSource {
    std::size_t layer_index;
    const MMapLlamaBlockWeights* weights;
};

// A stack of LLaMA blocks (RMSNorm -> GQA -> residual -> RMSNorm -> SwiGLU -> residual) whose
// weights and KV cache live on the GPU. No embedding, final norm or lm_head: the input is the
// hidden state of one token, the output is the hidden state after the last block.
//
// Weights are copied once, at construction, from the mmap'd float32 files (same column-major
// layout, no conversion). The KV cache is allocated on the device only.
class GpuTransformer {
public:
    // `capacity` = maximum number of tokens (prompt + generated) the KV cache can hold.
    GpuTransformer(const GpuModelConfig& config, const std::vector<GpuLayerSource>& layers,
                   std::size_t capacity)
        : config_{config},
          cache_{layers.size(), config.n_kv_heads * config.head_dim, capacity} {
        if (layers.empty()) throw std::invalid_argument("GpuTransformer: no layers");
        if (config_.n_heads * config_.head_dim != config_.d_model) {
            throw std::invalid_argument("GpuTransformer: d_model != n_heads * head_dim");
        }
        d_ff_ = layers.front().weights->gate_proj.out_features();

        for (const auto& layer : layers) check_shapes(*layer.weights);
        reserve_check(layers.size());

        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        blocks_.reserve(layers.size());
        for (const auto& layer : layers) {
            blocks_.push_back(upload_block(*layer.weights, layer.layer_index));
        }
        allocate_workspace();
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    ~GpuTransformer() {
        if (stream_) {
            cudaStreamSynchronize(stream_);
            cudaStreamDestroy(stream_);
        }
    }
    GpuTransformer(const GpuTransformer&) = delete;
    GpuTransformer& operator=(const GpuTransformer&) = delete;

    std::size_t n_layers() const { return blocks_.size(); }
    std::size_t d_model() const { return config_.d_model; }
    GpuKVCache& cache() { return cache_; }
    const GpuKVCache& cache() const { return cache_; }

    // One decode step, host to host. `hidden_in` is the (d_model) hidden state of the token at
    // absolute position `pos`; rows [0, pos) of the device cache must already hold the earlier
    // tokens. The token's K/V are written to row `pos` of every layer; `hidden_out` receives the
    // last block's output. The caller tracks the length (see GpuKVCache).
    void decode(const float* hidden_in, float* hidden_out, std::size_t pos) {
        const float* result = decode_from_host(hidden_in, pos);
        CUDA_CHECK(cudaMemcpyAsync(hidden_out, result, d_model_bytes(), cudaMemcpyDeviceToHost,
                                   stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    // Host in, device out: copies the hidden state up, runs the blocks, and returns without
    // waiting. The result (device pointer into an internal buffer, valid until the next call)
    // can feed more work on stream() before anything comes back to the host.
    const float* decode_from_host(const float* hidden_in, std::size_t pos) {
        CUDA_CHECK(cudaMemcpyAsync(x_[0].data(), hidden_in, d_model_bytes(),
                                   cudaMemcpyHostToDevice, stream_));
        return run_blocks(x_[0].data(), pos);
    }

    // Same step with the activation already on the device (what a CPU->GPU pipeline will use
    // after one host->device copy at the stage boundary). Returns a device pointer into an
    // internal buffer, valid until the next call; wait with synchronize() before reading it.
    const float* decode_device(const float* device_hidden_in, std::size_t pos) {
        return run_blocks(device_hidden_in, pos);
    }

    void synchronize() { CUDA_CHECK(cudaStreamSynchronize(stream_)); }
    cudaStream_t stream() const { return stream_; }

private:
    struct BlockBuffers {
        DeviceBuffer<float> attn_norm, ffn_norm;
        DeviceBuffer<float> q, k, v, o;
        DeviceBuffer<float> gate, up, down;
        bool use_rope = true;
    };

    std::size_t d_model_bytes() const { return config_.d_model * sizeof(float); }
    std::size_t q_dim() const { return config_.n_heads * config_.head_dim; }
    std::size_t kv_dim() const { return config_.n_kv_heads * config_.head_dim; }

    void check_shapes(const MMapLlamaBlockWeights& w) const {
        const std::size_t d = config_.d_model;
        const bool ok = w.input_layernorm.in_features() == d &&
                        w.post_attention_layernorm.in_features() == d &&
                        w.q_proj.in_features() == d && w.q_proj.out_features() == q_dim() &&
                        w.k_proj.in_features() == d && w.k_proj.out_features() == kv_dim() &&
                        w.v_proj.in_features() == d && w.v_proj.out_features() == kv_dim() &&
                        w.o_proj.in_features() == q_dim() && w.o_proj.out_features() == d &&
                        w.gate_proj.in_features() == d && w.gate_proj.out_features() == d_ff_ &&
                        w.up_proj.in_features() == d && w.up_proj.out_features() == d_ff_ &&
                        w.down_proj.in_features() == d_ff_ && w.down_proj.out_features() == d;
        if (!ok) throw std::invalid_argument("GpuTransformer: block weight shapes do not match");
    }

    // Fail early, with numbers, instead of a bare cudaErrorMemoryAllocation half way through.
    // The KV cache is already allocated at this point, so only the weights are counted.
    void reserve_check(std::size_t n_layers) const {
        const std::size_t d = config_.d_model;
        const std::size_t block_floats =
            2 * d + 2 * d * q_dim() + 2 * d * kv_dim() + 3 * d * d_ff_;
        const std::size_t need = n_layers * block_floats * sizeof(float);
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        if (need > free_bytes) {
            throw std::runtime_error(
                "GpuTransformer: " + std::to_string(n_layers) + " layers need " +
                std::to_string(need >> 20) + " MiB of weights but only " +
                std::to_string(free_bytes >> 20) + " MiB of GPU memory are free");
        }
    }

    static DeviceBuffer<float> upload_weight(const MMapWeight& w) {
        DeviceBuffer<float> buffer(w.in_features() * w.out_features());
        buffer.upload(w.data(), buffer.size());
        return buffer;
    }

    BlockBuffers upload_block(const MMapLlamaBlockWeights& w, std::size_t layer_index) {
        BlockBuffers b;
        b.attn_norm = upload_weight(w.input_layernorm);
        b.ffn_norm = upload_weight(w.post_attention_layernorm);
        b.q = upload_weight(w.q_proj);
        b.k = upload_weight(w.k_proj);
        b.v = upload_weight(w.v_proj);
        b.o = upload_weight(w.o_proj);
        b.gate = upload_weight(w.gate_proj);
        b.up = upload_weight(w.up_proj);
        b.down = upload_weight(w.down_proj);
        b.use_rope = (layer_index % 4 != 3);  // NoPE every 4th layer, as in SmolLM3
        return b;
    }

    void allocate_workspace() {
        x_[0] = DeviceBuffer<float>(config_.d_model);
        x_[1] = DeviceBuffer<float>(config_.d_model);
        h_ = DeviceBuffer<float>(config_.d_model);
        xn_ = DeviceBuffer<float>(config_.d_model);
        q_ = DeviceBuffer<float>(q_dim());
        k_ = DeviceBuffer<float>(kv_dim());
        v_ = DeviceBuffer<float>(kv_dim());
        attn_ = DeviceBuffer<float>(q_dim());
        act_ = DeviceBuffer<float>(d_ff_);
    }

    // Runs every block on stream_ and returns the device pointer holding the result.
    const float* run_blocks(const float* input, std::size_t pos) {
        if (pos + 1 > cache_.capacity()) {
            throw std::length_error("GpuTransformer: KV cache capacity exceeded");
        }
        const int d = static_cast<int>(config_.d_model);
        const int heads = static_cast<int>(config_.n_heads);
        const int kv_heads = static_cast<int>(config_.n_kv_heads);
        const int hd = static_cast<int>(config_.head_dim);
        const int ff = static_cast<int>(d_ff_);
        const int p = static_cast<int>(pos);

        const float* in = input;
        float* out = x_[0].data() == input ? x_[1].data() : x_[0].data();
        for (std::size_t i = 0; i < blocks_.size(); ++i) {
            const BlockBuffers& b = blocks_[i];

            rmsnorm(in, b.attn_norm.data(), xn_.data(), d, config_.rms_eps, stream_);
            gemv(xn_.data(), b.q.data(), nullptr, q_.data(), d, heads * hd, stream_);
            gemv(xn_.data(), b.k.data(), nullptr, k_.data(), d, kv_heads * hd, stream_);
            gemv(xn_.data(), b.v.data(), nullptr, v_.data(), d, kv_heads * hd, stream_);
            rope_append_kv(q_.data(), k_.data(), v_.data(), cache_.k(i), cache_.v(i), p, heads,
                           kv_heads, hd, config_.rope_theta, b.use_rope, stream_);
            attention_decode(q_.data(), cache_.k(i), cache_.v(i), attn_.data(), p + 1, heads,
                             kv_heads, hd, stream_);
            gemv(attn_.data(), b.o.data(), in, h_.data(), heads * hd, d, stream_);  // + x

            rmsnorm(h_.data(), b.ffn_norm.data(), xn_.data(), d, config_.rms_eps, stream_);
            gemv_swiglu(xn_.data(), b.gate.data(), b.up.data(), act_.data(), d, ff, stream_);
            gemv(act_.data(), b.down.data(), h_.data(), out, ff, d, stream_);  // + h

            in = out;
            out = (out == x_[0].data()) ? x_[1].data() : x_[0].data();
        }
        return in;
    }

    GpuModelConfig config_;
    std::size_t d_ff_ = 0;
    GpuKVCache cache_;
    cudaStream_t stream_ = nullptr;
    std::vector<BlockBuffers> blocks_;

    // Workspace shared by all blocks (they run one after another). x_ ping-pongs between blocks.
    DeviceBuffer<float> x_[2];
    DeviceBuffer<float> h_, xn_, q_, k_, v_, attn_, act_;
};

}  // namespace gpu
