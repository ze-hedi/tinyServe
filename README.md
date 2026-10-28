# tinyServe

A small C++20 CPU inference runtime for [SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B), with custom matrix kernels, memory-mapped FP32 weights, and interactive terminal chat.

tinyServe implements the inference pipeline directly: tokenization, chat formatting, transformer blocks, KV caching, and greedy decoding. It is an experimental runtime and performance playground. The current entry points are command-line programs and a terminal UI; there is no HTTP server.

## Features

- **Full-model inference:** 36 transformer layers, grouped-query attention, RMSNorm, SwiGLU, and a tied embedding/output projection.
- **Memory-mapped weights:** read-only FP32 files accessed without copying the full model into owned weight buffers.
- **CPU kernels:** runtime-dispatched AVX2/FMA kernels with scalar fallbacks, separate prefill and single-token decode paths, and a persistent worker pool.
- **Multi-turn chat:** streamed output, configurable system prompts, thinking-mode formatting, and reuse of the cached conversation prefix.
- **Native tokenizer:** byte-level BPE with PCRE2 pre-tokenization and support for added/special tokens.
- **Validation tools:** unit tests, Hugging Face chat-template golden fixtures, and mapped-versus-owned numerical comparisons.

## Quick start

Run the commands below from the repository root. The default model path, `smollm3`, is relative to the current working directory.

### 1. Requirements

- A little-endian Linux host is the primary target. The loader uses POSIX `mmap`; terminal handling and CPU-affinity code are also platform-specific.
- CMake **3.20+**, Git, a C/C++ toolchain with **C++20** support, and Make or Ninja. GCC is a suitable starting point; the code uses GNU-style compiler attributes and flags.
- Python **3.9+** for model preparation, plus `torch`, `numpy`, `safetensors`, and `huggingface_hub`.
- Network access for the initial dependency and model downloads.

CMake fetches **PCRE2 10.44** and **GoogleTest 1.15.2** automatically. The JSON library is vendored in `include/json.hpp`. Inference uses the C++ runtime; PyTorch is needed only for offline weight conversion.

Plan for roughly **12.3 GB of exported FP32 weights**, in addition to the downloaded BF16 checkpoint (roughly 6.2 GB). The default 4096-token FP32 KV cache uses **576 MiB**, plus activations and other runtime allocations. Memory mapping avoids a second owned copy of the weights, but resident pages still consume RAM; performance depends heavily on available RAM and memory bandwidth.

### 2. Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
```

For host-specific compilation, add `-DTINYSERVE_NATIVE=ON` to the configure command. This adds `-march=native` to the mmap, generation, and TUI targets. The default build already dispatches to AVX2/FMA kernels on compatible x86 CPUs.

### 3. Download and convert SmolLM3

Install the Python dependencies in your preferred Python environment:

```bash
python3 -m pip install torch numpy safetensors huggingface_hub
```

Download the checkpoint and tokenizer into the cache location expected by the converter:

```bash
python3 -c 'from pathlib import Path; from huggingface_hub import snapshot_download; snapshot_download(repo_id="HuggingFaceTB/SmolLM3-3B", cache_dir=str(Path.home() / ".cache/huggingface/hub"), allow_patterns=["*.safetensors", "*.json"])'
python3 convert_first_block.py
```

Despite its name, `convert_first_block.py` exports **the entire model**. It finds the most recently modified snapshot under `~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM3-3B`, reads the sharded BF16 checkpoint, converts tensors to FP32, and writes `smollm3/`. The script currently has no command-line options and uses that fixed cache path rather than `HF_HOME`.

Expected output:

```text
smollm3/
├── embed_tokens.bin
├── embed_tokens.json
├── final_norm.bin
├── final_norm.json
├── bloc_0/
│   ├── shapes.json
│   ├── input_layernorm.bin
│   ├── q_proj.bin, k_proj.bin, v_proj.bin, o_proj.bin
│   ├── post_attention_layernorm.bin
│   └── gate_proj.bin, up_proj.bin, down_proj.bin
├── ...
├── bloc_35/
└── tokenizer/
    ├── tokenizer.json
    └── tokenizer_config.json
```

The converter copies tokenizer files if they are present in the snapshot. `tokenizer/tokenizer.json` is required by the runtime. Generated `smollm3/` weights are ignored by Git.

### 4. Chat

Simple line-oriented chat:

```bash
./build/bench_generate smollm3 --no-think --max-new 128
```

Terminal UI with a dedicated input box:

```bash
./build/tui/tui_chat smollm3 --no_think
```

The flag spelling differs: **`--no-think`** for `bench_generate`, **`--no_think`** for `tui_chat`. Both enable thinking mode by default.

## Chat options and controls

| Option | `bench_generate` | `tui_chat` | Meaning |
|---|---|---|---|
| Positional model directory | Yes | Yes | Defaults to `smollm3` |
| `--system TEXT` | Yes | Yes | Defaults to `You are a helpful assistant.` |
| `--threads N` | Yes | Yes | Worker count; defaults to min(6, hardware concurrency), with a fallback of 1 |
| `--ctx N` | Yes | Yes | Total prompt + generation capacity; default 4096 tokens |
| `--layers N` | Yes | Yes | Use the first N layers for debugging; 0 means all 36 |
| `--max-new N` | Yes | No | Per-turn generation limit; otherwise uses remaining context capacity |
| `--think` / `--no-think` | Yes | No | Enable/disable thinking-mode prompt formatting |
| `--no_think` | No | Yes | Disable thinking-mode prompt formatting |
| `--verify` | Yes | No | Recompute the final generated-token prediction without the cache and compare argmax |

Partial-layer runs are useful for debugging and timing, but do not represent normal model output.

In either chat interface:

- `/reset` clears conversation history and keeps the system prompt.
- `/think` and `/no_think` change the prompt's thinking mode.
- `/exit` exits.

The line-oriented interface also supports `/system TEXT` (replace the system prompt and clear history), `/show` (print tagged context), `/help`, and `/quit`.

In `bench_generate`, Ctrl-C stops generation and exits when pressed at the prompt. In the TUI, Ctrl-C or Esc stops an answer. Cancellation is checked between generated tokens, so it does not interrupt an in-progress forward pass.

Generation uses greedy argmax and stops at `<|im_end|>`, the token limit, or cancellation. Context does not automatically slide or evict old messages: use `/reset` when it fills. The line-oriented program currently labels every token-limit stop `[context full]`, including stops caused by `--max-new`.

## How inference works

```text
Message history (SmollAgent)
  → SmolLM3 chat template
  → BPE token IDs
  → Embedding lookup
  → 36 × [RMSNorm → GQA → residual → RMSNorm → SwiGLU → residual]
  → Final RMSNorm
  → Tied embedding matrix as LM head
  → Greedy next token
```

### Model and attention

`SmolLM3<float>` owns the mapped weights, tokenizer, transformer blocks, and thread pool. Its default architecture uses a hidden size of 2048, 16 query heads, 4 KV heads, and a head dimension of 128. RoPE uses theta 5,000,000; every fourth layer (indices 3, 7, …) omits positional rotation, matching the model's 3:1 RoPE/NoPE pattern.

The runtime does not load architecture settings from Hugging Face `config.json`. Defaults live in `SmolLM3::Config`, while tensor dimensions come from the exported manifests. In particular, the actual output vocabulary comes from the embedding matrix (128256 entries in the SmolLM3 export); the `Config::vocab_size` field currently has a stale default of 49152 and does not determine the projection size.

### Prefill, decode, and cache reuse

- **Prefill** processes new prompt tokens and stores each layer's keys and values. An initial prompt uses the matrix-multiplication attention path.
- **Decode** feeds one generated token back at a time, attending directly over cached K/V buffers. Keys are stored after RoPE where applicable.
- **Across turns**, `SmollSession` renders and tokenizes the updated history, finds its longest shared prefix with the cached token IDs, truncates the cache to that prefix, and prefills only the remaining suffix. Earlier edits or mode changes can reduce reuse.
- Only the last hidden row is normalized and projected when requesting next-token logits. `forward()` instead returns logits for every input position.

The cache allocates capacity up front on first generation. Its storage is `2 × layers × context_tokens × kv_heads × head_dim × sizeof(T)` bytes.

### Weight layout and compute

The current export format is **`tinyserve-float32-matrices`, version 2**: little-endian FP32, column-major tensors, and JSON manifests containing shapes and byte sizes. Version 1 exports must be regenerated.

Linear weights are logically `[in, out]`, so inference computes `X @ W`. A Hugging Face `[out, in]` row-major tensor already has the correct byte order for that column-major interpretation; conversion changes precision without transposing the bytes. Embeddings are `[hidden, vocab]`, norm vectors are `[hidden, 1]`, and activations in `Mat<T>` are row-major. The embedding mapping also serves as the output projection, so there is no separate `lm_head.bin`.

The FP32 mapped path uses tiled 2×4 AVX2/FMA micro-kernels for prefill and a 1×4 kernel for single-token decode. Decode output columns are distributed dynamically in 16-column units. A persistent fork/join pool avoids per-operation thread creation, with best-effort Linux CPU pinning. Tune `--threads` on your machine rather than assuming more workers are faster.

### Using the headers

The runtime is implemented primarily in headers. A minimal conversation looks like:

```cpp
#include "smoll_session.h"
#include <iostream>

int main() {
    SmolLM3<float> model("smollm3");
    SmollSession<float> session(model);
    session.set_context_tokens(4096);  // before the cache is created
    session.agent().set_thinking(false)
                   .set_system_prompt("Answer briefly.")
                   .add_user("What is the capital of France?");

    Decoder<float> decoder(DecodingStrategy::Greedy);
    auto result = session.generate(decoder, 64);
    std::cout << result.text << '\n';
    session.agent().add_assistant(result.text);
}
```

Add `include/` to your target's include path and link `pcre2-8-static`, following the existing CMake targets. `generate()` returns text, token IDs, stop reason, cache-reuse counts, and timing; it does **not** append the answer to history automatically. Its optional callback receives token pieces and can return `false` to cancel. Individual pieces may contain partial UTF-8 characters; `result.text` is the complete decoding.

Model execution is intended to be sequential: the thread pool's `run()` contract permits one caller at a time, with no nested calls. Tool descriptions and results can be formatted through `SmollAgent`, but there is no tool-execution engine.

## Tests

After preparing the model/tokenizer:

```bash
ctest --test-dir build --output-on-failure
```

Six suites are registered: matrix operations, embeddings, grouped-query attention, transformer blocks, KV caching, and chat templates/tokenizer integration. The core tests use synthetic weights. `test_chat_template` additionally requires `smollm3/tokenizer/tokenizer.json` at the source-tree location and compares prompts and token IDs against committed Hugging Face golden fixtures.

To run the core suites before downloading a tokenizer:

```bash
ctest --test-dir build --output-on-failure -E test_chat_template
```

For the full suite without downloading model weights, fetch just the tokenizer:

```bash
python3 -m pip install huggingface_hub
python3 -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="HuggingFaceTB/SmolLM3-3B", allow_patterns=["tokenizer.json", "tokenizer_config.json"], local_dir="smollm3/tokenizer")'
ctest --test-dir build --output-on-failure
```

The standalone `tokenizer_test` executable is separate from CTest. Override its machine-specific fallback path explicitly:

```bash
TINY_SERVE_TOKENIZER_JSON=smollm3/tokenizer/tokenizer.json ./build/tokenizer_test
```

`tests/test_tokenizer.cpp` is not enabled as a CMake target. To regenerate the chat golden fixtures, install `transformers`, ensure the SmolLM3 tokenizer is in the Hugging Face cache, then run `HF_HUB_OFFLINE=1 python3 tools/gen_chat_golden.py`.

## Benchmarks and inspection tools

| Executable (under `build/`) | Purpose |
|---|---|
| `bench_block` | Owned-weight transformer block with synthetic weights and input; no model download needed |
| `bench_block_mmap` | One exported block; timing, page-fault counts, and optional owned-reference comparison |
| `bench_smollm3_mmap` | Full-model forward on synthetic token IDs; optional partial layers and reference check |
| `bench_generate_next` | One next-token prediction from raw text, without chat formatting |
| `bench_chat_prefill` | Rendered chat prompt → first token; inspect prompt, IDs, top logits, and repeated prefill timings |
| `bench_generate` | Interactive multi-turn generation with cache-reuse and decode timing statistics |
| `example_chat_context` | Render and tokenize message history without loading model weights |
| `tokenizer_test` | Standalone tokenizer throughput benchmark |
| `tui/tui_chat` | Terminal chat interface |

Examples:

```bash
# Inspect tagged context and token IDs (tokenizer only).
./build/example_chat_context --user "Hi" --ids

# Predict one token from an unformatted completion prompt.
./build/bench_generate_next smollm3 --prompt "The capital of France is" --threads 6

# Measure chat prefill and inspect the first-token candidates.
./build/bench_chat_prefill smollm3 --user "What is the capital of France?" \
  --threads 6 --show-prompt --top 5 --repeat 3

# Compare an exported block against the owned-weight implementation.
./build/bench_block_mmap smollm3/bloc_0 \
  --seq 16 --threads 6 --warmup 1 --iters 3 --check

# Small full-model-forward benchmark restricted to one layer.
./build/bench_smollm3_mmap smollm3 \
  --seq 4 --layers 1 --threads 6 --warmup 1 --iters 3 --check

# Bound generation length and compare cached vs. uncached final-token predictions.
./build/bench_generate smollm3 --no-think --max-new 16 --verify
```

Both mmap benchmarks accept `--prefault` (`MAP_POPULATE`) and `--hugepages` (`MADV_HUGEPAGE` advice), as well as `--seq`, `--threads`, `--warmup`, `--iters`, and `--check`. Their `--help` prints usage. Passing `smollm3/bloc_0` explicitly avoids the single-block benchmark's legacy default of `bloc_0`.

For first-touch measurements, include mapping/load time as well as the first forward. Warm page-cache results are different from storage-cold results. A one-token block benchmark is also different from end-to-end cached generation. Reference checks allocate owned weight copies and can substantially increase memory use.

**Current validation status:** while preparing this README, the Release build and all six CTest suites passed, as did the tokenizer benchmark and context-inspection example. However, the two `--check` examples above reported `mmap output differs from the non-mmap reference` with the local SmolLM3 export (single block at sequence length 16; one-layer model at sequence length 4). These numerical mismatches remain unresolved; passing unit tests alone does not establish full-model numerical equivalence.

## Code map

| Path | Responsibility |
|---|---|
| `include/smollm3.h` | Model assembly, forward passes, cache creation, tied LM head |
| `include/smoll_session.h` | Generation loop, stop conditions, multi-turn prefix reuse |
| `include/smoll_agent.h`, `include/chat_template.h` | Message history and SmolLM3 prompt rendering |
| `include/tokenizer.h`, `include/decoder.h` | Byte-level BPE and greedy token selection |
| `include/llama_mmap_block.h` | Mapped-weight attention, MLP, normalization, transformer block |
| `include/mmap_block_loader.h`, `include/embeddings_mmap.h` | Validated read-only mappings and embedding lookup |
| `include/mat.h`, `include/micro_kernels/` | Matrix operations and CPU kernels |
| `include/thread_pool.h`, `include/kv_cache.h` | Persistent workers and per-layer K/V storage |
| `include/llama_transformer.h`, `include/embeddings.h` | Owned-weight implementations used in tests and comparisons |
| `include/safetensors_loader.h` | Separate BF16 first-block loader; not the full-model loading path |
| `convert_first_block.py` | Full SmolLM3 BF16 → FP32 export |
| `download_qwen.py` | Standalone Qwen download/FP16 conversion utility; its output is not supported by the SmolLM3 runtime |
| `tui/` | Terminal control, input widget, interactive chat |
| `tests/`, `tools/gen_chat_golden.py` | Correctness tests and upstream-template fixture generation |

## Current scope and troubleshooting

- **Model support:** the end-to-end path targets SmolLM3-3B and its custom export format. Arbitrary Hugging Face checkpoints, FP16/quantized inference, GPU execution, sampling strategies beyond greedy decoding, and batched request serving are not implemented.
- **Missing model or manifest:** run the converter from the repository root, or supply the exported model directory explicitly. The runtime cannot load a raw Hugging Face snapshot as its model root.
- **Unsupported manifest format:** regenerate weights with the current converter; the loader requires version 2 column-major exports.
- **Tokenizer file missing:** make sure `tokenizer.json` was downloaded and copied into `smollm3/tokenizer/`. The tests use this source-tree path even when applications use a different model root.
- **Context full:** use `/reset`, lower the requested response length, or start with a larger `--ctx`. Larger contexts allocate more KV memory and make long-prompt attention more expensive.
- **Slow generation:** compare thread counts, distinguish first-touch from steady-state timing, and check that the model's working set fits in available RAM. FP32 decoding streams substantial weight data for every token.

## Performance notes

- [FP32 decode optimization](FP32_DECODE_OPTIMIZATION.md) records kernel-width and scheduling experiments for full-model generation.
- [LLaMA mmap performance experiments](LLAMA_MMAP_PERFORMANCE.md) records earlier single-block optimization work.

These are historical experiment reports, not portable throughput guarantees. In particular, the older mmap report describes a row-major export and an implementation without persistent workers or KV caching; the current runtime uses the version 2 column-major layout, a persistent pool, and cached decoding described above.
