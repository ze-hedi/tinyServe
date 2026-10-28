#pragma once

#include "tail_backend.h"

#include <cstddef>
#include <memory>

template <typename T>
class SmolLM3;

namespace gpu {

// Builds a TailBackend that runs the model's last `gpu_layers` transformer blocks (and, with
// `with_head`, the final norm + lm_head) on the GPU, uploading their weights from the model's
// own mmap'd tensors. `kv_capacity` is the GPU KV cache size in tokens; it must be at least the
// context length of the session that will use the model. Throws if the GPU lacks the memory.
// Attach the result with SmolLM3::set_tail.
std::unique_ptr<TailBackend> make_gpu_tail(const SmolLM3<float>& model, std::size_t gpu_layers,
                                           bool with_head, std::size_t kv_capacity);

}  // namespace gpu
