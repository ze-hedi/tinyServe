#include "gpu_tail.h"

#include "gpu_head.h"
#include "gpu_transformer.h"
#include "smollm3.h"

#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpu {
namespace {

class GpuTail final : public TailBackend {
public:
    GpuTail(const GpuModelConfig& config, std::size_t first_layer,
            const std::vector<GpuLayerSource>& layers, const MMapWeight* final_norm,
            const MMapWeight* lm_head, std::size_t kv_capacity)
        : first_layer_{first_layer},
          kv_capacity_{kv_capacity},
          transformer_{config, layers, kv_capacity} {
        if (final_norm && lm_head) {
            // Same stream as the blocks: the head follows them without a host sync in between.
            head_ = std::make_unique<GpuHead>(*final_norm, *lm_head, config.rms_eps,
                                              transformer_.stream());
        }
        description_ = "GPU tail: layers " + std::to_string(first_layer_) + "-" +
                       std::to_string(first_layer_ + layers.size() - 1) +
                       (head_ ? " + final norm + lm_head" : " (head stays on the CPU)");
    }

    std::size_t first_layer() const override { return first_layer_; }
    bool has_head() const override { return head_ != nullptr; }
    std::size_t kv_capacity() const override { return kv_capacity_; }
    std::string description() const override { return description_; }

    void upload_kv(std::size_t layer, std::size_t row0, std::size_t rows, const float* k,
                   const float* v) override {
        transformer_.cache().upload_rows(local(layer), row0, rows, k, v);
    }
    void download_kv(std::size_t layer, std::size_t row0, std::size_t rows, float* k,
                     float* v) override {
        transformer_.cache().download_rows(local(layer), row0, rows, k, v);
    }

    void decode(const float* hidden_in, float* hidden_out, std::size_t pos) override {
        transformer_.decode(hidden_in, hidden_out, pos);
    }

    void decode_logits(const float* hidden_in, std::size_t pos, float* logits_out) override {
        require_head();
        const float* hidden = transformer_.decode_from_host(hidden_in, pos);
        const float* logits = head_->logits_device(hidden);
        CUDA_CHECK(cudaMemcpyAsync(logits_out, logits, head_->vocab_size() * sizeof(float),
                                   cudaMemcpyDeviceToHost, transformer_.stream()));
        transformer_.synchronize();
    }

    void head_logits(const float* hidden, float* logits_out) override {
        require_head();
        head_->logits(hidden, logits_out);
    }

private:
    std::size_t local(std::size_t layer) const {
        if (layer < first_layer_ || layer - first_layer_ >= transformer_.n_layers()) {
            throw std::out_of_range("GpuTail: layer " + std::to_string(layer) +
                                    " is not a tail layer");
        }
        return layer - first_layer_;
    }
    void require_head() const {
        if (!head_) throw std::logic_error("GpuTail: the head is not on the GPU");
    }

    std::size_t first_layer_;
    std::size_t kv_capacity_;
    GpuTransformer transformer_;
    std::unique_ptr<GpuHead> head_;
    std::string description_;
};

}  // namespace

std::unique_ptr<TailBackend> make_gpu_tail(const SmolLM3<float>& model, std::size_t gpu_layers,
                                           bool with_head, std::size_t kv_capacity) {
    const auto& cfg = model.config();
    if (gpu_layers == 0 || gpu_layers > cfg.n_layers) {
        throw std::invalid_argument("make_gpu_tail: gpu_layers must be in [1, " +
                                    std::to_string(cfg.n_layers) + "]");
    }
    if (kv_capacity == 0) throw std::invalid_argument("make_gpu_tail: kv_capacity is zero");

    GpuModelConfig config;
    config.n_heads = cfg.n_heads;
    config.n_kv_heads = cfg.n_kv_heads;
    config.head_dim = cfg.head_dim;
    config.d_model = cfg.d_model;
    config.rope_theta = cfg.rope_theta;

    const std::size_t first = cfg.n_layers - gpu_layers;
    std::vector<GpuLayerSource> sources;
    sources.reserve(gpu_layers);
    for (std::size_t i = first; i < cfg.n_layers; ++i) {
        sources.push_back({i, &model.block_weights(i)});
    }

    return std::make_unique<GpuTail>(config, first, sources,
                                     with_head ? &model.final_norm_weights() : nullptr,
                                     with_head ? &model.embed_weights() : nullptr, kv_capacity);
}

}  // namespace gpu
