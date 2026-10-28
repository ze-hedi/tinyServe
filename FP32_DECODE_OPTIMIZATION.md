# FP32 Decode Optimization: Step by Step

## Goal

The goal was to make SmolLM3 generate tokens faster while keeping **FP32 weights and FP32 computation**.

The initial target was to reach approximately **2 tokens/second**.

The final result was a median of **2.27 tokens/second**, with individual runs reaching **2.73 tokens/second**.

---

## Step 1: Understand the existing code

I first explored the project with several parallel analysis agents. Each agent examined a different part of the runtime:

- FP32 matrix multiplication and AVX2 kernels.
- Thread creation, pinning, and work distribution.
- Attention and KV-cache operations.
- Build flags and hardware behavior.
- Benchmark and correctness-test coverage.

The main decode operation is a vector-matrix multiplication:

```text
one activation row × one FP32 weight matrix
```

This operation runs for every attention and MLP projection in every transformer layer. The tied embedding matrix is also used as the final language-model head.

The exploration showed that decode is mainly limited by reading FP32 weights from memory. Each generated token reads approximately **12.3 GB of weights**.

---

## Step 2: Reproduce the starting performance

I rebuilt the existing code and measured generation before changing the FP32 kernel.

The original default configuration used 20 workers. On the fixed benchmark, it produced approximately:

```text
0.96–0.98 tokens/second
```

Restricting execution to six physical performance cores improved the early result to approximately:

```text
1.17 tokens/second
```

This showed that using every logical CPU was not the best choice on this hybrid Intel processor.

---

## Step 3: Make benchmarking repeatable

The interactive benchmark originally generated until the model ended its answer. That made different runs difficult to compare.

I added two options to `chat.cpp`:

```text
--no-think
--max-new N
```

`--no-think` disables reasoning mode. `--max-new N` stops generation after a fixed number of tokens.

The main comparison used the same prompt and 16 generated tokens for every run:

```bash
./build/chat --no-think --max-new 16
```

---

## Step 4: Change static work distribution to dynamic distribution

The decode matrix multiplication divides the output columns between workers.

Originally, each worker received one large fixed range of columns. This could leave faster workers waiting for slower workers.

I changed the implementation to divide the output into small **16-column units**. Workers claim these units dynamically through the existing thread pool.

This means:

- Every output column is still computed exactly once.
- Workers never write to the same output unit.
- Faster cores can process more units.
- The FP32 arithmetic does not change.

Using six physical performance cores, this experiment reached approximately:

```text
1.66 tokens/second
```

For comparison, static distribution with the same six cores reached approximately:

```text
1.42 tokens/second
```

Dynamic distribution was therefore kept.

---

## Step 5: Test different FP32 micro-kernel widths

The decode kernel processes several output columns at the same time.

The original implementation used a **1×8 micro-kernel**: one activation row and eight output columns. This creates eight independent FP32 weight streams per task.

I tested other widths without changing the weight format or arithmetic type.

### 1×12 kernel

This reduced activation loads but created too many weight streams. It reached approximately:

```text
1.50 tokens/second
```

It was rejected.

### 1×4 kernel

This used four output columns and four weight streams per task. It reached approximately:

```text
1.96 tokens/second
```

The result suggested that four streams were easier for the CPU memory and prefetch systems to handle than eight streams.

The 1×4 FP32 AVX2/FMA kernel was kept.

### 1×2 kernel

I also tested two output columns. It reached approximately:

```text
1.60 tokens/second
```

It was slower than 1×4, so it was rejected.

---

## Step 6: Improve CPU affinity

The machine has performance cores, efficiency cores, and SMT siblings.

Using 20 workers caused slower workers and extra synchronization to limit each projection. The best measurements used six workers on physical performance cores.

I integrated this behavior into the runtime:

1. The default generation thread count is now six, capped by available hardware concurrency.
2. Allowed CPUs are ordered to select one SMT sibling per core before selecting the second siblings.
3. Workers are pinned using this ordering.
4. The calling thread is also pinned to reduce migration and wake-up overhead.

After integrating caller affinity, a run without manual `taskset` reached:

```text
2.39 tokens/second
```

The optimization therefore works with the normal application defaults.

---

## Step 7: Verify numerical correctness

The optimizations change scheduling and kernel width, but they do not change the FP32 weight format.

Validation included:

- Building all project targets.
- Running all six release test suites.
- Comparing cached generation with an uncached recomputation.
- Comparing a mapped model layer with the ordinary reference implementation.
- Running core math and KV-cache tests with AddressSanitizer and UndefinedBehaviorSanitizer.

Results:

```text
6/6 release test suites passed
Cached vs. uncached argmax: MATCH
One-layer differential reference: passed
Core math and KV-cache sanitizer tests: passed
```

The tokenizer release test also passed. Its sanitizer build encountered a recursive AddressSanitizer runtime failure in the tokenizer/PCRE2 path.

---

## Step 8: Measure the final version

I ran the final benchmark three times with default settings and no manual CPU pinning.

| Run | Time per token | Tokens/second |
|---|---:|---:|
| 1 | 440 ms | 2.27 |
| 2 | 445 ms | 2.25 |
| 3 | 366 ms | 2.73 |

The median result was:

```text
2.27 tokens/second
```

This exceeds the original 2 tokens/second target.

---

## Files involved in the final optimization

- `include/mat.h`
  - Dynamic 16-column decode work units.
  - 1×4 FP32 decode micro-kernel selection.
  - Six-worker default helper.
- `include/thread_pool.h`
  - SMT-aware CPU ordering.
  - Caller and worker affinity.
- `include/smollm3.h`
  - Uses the new default thread-count helper.
- `tui/main.cpp`
  - Uses the new default thread-count helper.
- `chat.cpp`
  - Adds repeatable benchmark options and uses the new default.

## Reproduce the result

```bash
cmake --build build --target chat -j2
./build/chat --no-think --max-new 16
```

All accepted optimizations remain strictly **FP32-only**.
