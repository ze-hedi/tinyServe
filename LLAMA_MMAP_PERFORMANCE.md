# Optimizing `llama_mmap_block`: Kernel, Scheduling, Memory, and Benchmarking Experiments

## Abstract

This article documents a focused performance-engineering effort on the memory-mapped LLaMA transformer block in `tinyServe`. The work covered static architecture exploration, GEMM kernel optimization, thread scheduling, small-sequence parallelism, temporary-buffer reduction, RoPE reuse, memory-mapping policy experiments, benchmark infrastructure, and numerical validation.

The main controlled A/B experiment used the same FP32 weights, the same input, 20 worker threads, two warmups, and five measured iterations per run. Across five repeated processes, the optimized implementation reduced median latency from **340.4 ms to 205.5 ms** for a 200-token prefill-like workload and from **67.4 ms to 28.3 ms** for a one-token decode-like workload. These correspond to speedups of approximately **1.66×** and **2.38×**, respectively.

A separate first-touch experiment showed that prefaulting the mappings reduced median load-plus-first-forward latency for the one-token workload from approximately **227 ms to 30 ms** on a warm OS page cache. This is a page-fault and first-touch result, not a storage-cold I/O benchmark.

All work is on the branch:

```text
experiment/llama-mmap-performance
```

The changes are present in the working tree; no commit was created as part of this task.

---

## 1. Scope and Goals

The investigation was deliberately restricted to `llama_mmap_block` and the code directly required to execute, measure, and validate it. The ordinary in-memory block was used only as a numerical reference and, where appropriate, as an architectural comparison.

The goals were to:

1. Understand the mmap-backed block's complete execution path.
2. Identify performance bottlenecks using code inspection and controlled experiments.
3. Improve the FP32 matrix-multiplication kernel.
4. Improve parallelism for both large and small sequence lengths.
5. Reduce avoidable memory traffic and redundant computation.
6. Separate steady-state execution from mmap first-touch behavior.
7. Build a benchmark capable of supporting repeatable future experiments.
8. Validate every accepted transformation numerically and with sanitizers.

The work did **not** attempt to implement a complete production inference runtime, KV caching, quantization, or full-model execution.

---

## 2. Exploration Method

A multi-agent exploration was used to inspect the implementation from several independent perspectives:

- Compute kernels, loop nests, vectorization, and arithmetic intensity.
- Mmap ownership, file layout, alignment, page faults, TLB behavior, and temporary memory.
- Build flags, benchmark validity, and experimental design.
- Thread decomposition, synchronization, load balance, and allocation behavior.
- Existing correctness tests and independent reference paths.

After implementation, two additional review passes examined:

- Kernel correctness, tails, alignment, threading, and generic-type behavior.
- Loader RAII, benchmark semantics, CLI handling, test-fixture safety, and performance-claim validity.

This process identified the central fact that guided the rest of the work: **mmap affects weight residency and loading, but most steady-state performance problems were in the compute kernel and parallel schedule, not in `mmap()` itself.**

---

## 3. Workload and Measurement Environment

### 3.1 Hardware and software

The primary measurements were collected on:

| Property | Value |
|---|---|
| CPU | 13th Gen Intel Core i7-13700H |
| Logical CPUs exposed | 20 |
| Physical cores | 14 |
| NUMA nodes | 1 |
| L3 cache | 24 MiB |
| ISA used by optimized kernel | AVX2 and FMA |
| Compiler | GCC 11.4.0 |
| Language mode | C++20 |
| Baseline build | `-O3`, generic ISA |

The processor supports AVX2 and FMA, but the original benchmark did not enable `-march=native` or contain an explicit AVX2 kernel.

### 3.2 Model block

The mapped fixture contains layer 0 from SmolLM3-3B, converted offline from BF16 to contiguous FP32 matrices:

| Property | Value |
|---|---|
| Model dimension | 2048 |
| Query heads | 16 |
| KV heads | 4 |
| Head dimension | 128 |
| Feed-forward dimension | 11008 |
| Weight precision | FP32 |
| Total mapped weight size | Approximately 298.02 MiB |

The mapped linear weights are stored in `[in, out]` row-major form, matching `x @ W` in the C++ implementation.

### 3.3 Benchmark regimes

Two sequence lengths were emphasized:

- **`seq_len = 200`**: a one-block, full-sequence, prefill-like workload.
- **`seq_len = 1`**: a decode-like shape used to expose small-M parallelism and weight-bandwidth behavior.

The one-token case is **not** a complete cached-decoding benchmark. The block still executes its existing full-sequence API and does not implement a production KV-cache path.

---

## 4. Original Architecture

The original forward pass was:

```text
x
 ├─ RMSNorm
 ├─ Q/K/V projections from mmap weights
 ├─ grouped-query causal attention
 ├─ output projection from mmap weights
 ├─ residual addition
 ├─ RMSNorm
 ├─ gate/up projections from mmap weights
 ├─ SiLU and Hadamard product
 ├─ down projection from mmap weights
 └─ residual addition
```

The MLP projections dominate the arithmetic. At sequence length 200, the three large MLP GEMMs account for approximately **85.7% of GEMM FLOPs**, while all large weight-projection GEMMs account for approximately **99%**.

That distribution made projection quality the highest priority. Attention optimizations were still useful, but were not expected to dominate end-to-end prefill performance at this shape.

---

## 5. Problems Found in the Original Implementation

### 5.1 The intended register blocking was not reliably realized

The original micro-kernel used a runtime-sized `6 × 16` accumulator array:

```cpp
T acc[6][16];
```

The source comment assumed the compiler would map these values to registers. Inspection of the generated code showed that this was not guaranteed. The generic build used narrower SIMD instructions and stack-resident accumulator traffic in the hot loop.

The kernel therefore paid for both:

- Suboptimal instruction selection.
- Repeated accumulator loads and stores instead of true register residency.

### 5.2 Every row worker logically scanned the complete weight matrix

Parallelism was only over the M dimension. Each worker processed a different set of input rows but traversed all N and K tiles of the mapped weight matrix.

This had different consequences in the two regimes:

- At sequence 200, narrow per-worker row ranges reduced B reuse and created poor microtile shapes.
- At sequence 1, there was only one row, so each mapped projection ran with effectively one worker.

The one-token case therefore could not use additional cores to stream the approximately 298 MiB of weights.

### 5.3 The row partition systematically created one-row tails

At sequence 200, the original target of 30 threads produced:

```text
ceil(200 / 30) = 7 rows per worker
```

Most workers therefore executed:

```text
6-row microtile + 1-row microtile
```

The one-row tail reread B with very little reuse. A 20-thread configuration was already measurably better than 30 threads before the kernel rewrite, confirming that thread count and partition geometry mattered.

### 5.4 Threads were recreated for every operation

The original sequence-200 block executed 39 matrix multiplications, two RMSNorms, and one threaded activation. Static analysis estimated approximately:

```text
1,219 worker-thread constructions per forward pass
```

`threads_.reserve(30)` reused vector capacity, not operating-system threads.

A persistent pool remains a valuable future experiment, but the first implementation pass focused on reducing oversubscription and fixing the decomposition itself.

### 5.5 The MLP materialized unnecessary temporaries

The original MLP computed:

```text
gate      = x @ W_gate
up        = x @ W_up
activated = silu(gate)
fused     = activated * up
out       = fused @ W_down
```

At sequence 200, each `200 × 11008` FP32 matrix is approximately 8.4 MiB. Keeping `gate`, `up`, `activated`, and `fused` alive created approximately 33.6 MiB of large MLP intermediates, in addition to extra initialization and streaming passes.

### 5.6 RoPE repeated expensive work

The original attention loop recomputed `pow`, `sin`, and `cos` for each position/dimension pair in every head.

It also rotated K once per **query** head. With 16 query heads and four KV heads, each K head was rotated four times.

### 5.7 The benchmark could not support reliable optimization claims

The original benchmark:

- Ran exactly one forward pass.
- Had no warmup.
- Reported no distribution or repetition statistics.
- Did not distinguish first-touch from warm mappings.
- Used a fixed sequence length and fixed thread target.
- Printed only a few output values without a full numerical check.
- Included demand-page faults in the timed first forward without reporting them.

A single timing was useful as a rough observation, but not as a controlled optimization experiment.

---

## 6. Establishing a Better Baseline

The benchmark was extended before accepting the final optimizations.

It now supports:

```text
--seq N
--threads N
--warmup N
--iters N
--prefault
--hugepages
--check
```

It uses `std::chrono::steady_clock` and reports:

- Mapping/load time.
- Load-time minor and major faults.
- Individual warmup and measured iteration times.
- Per-iteration minor and major faults.
- A checksum consumed outside the timed region.
- Median, mean, minimum, and maximum latency.
- Output shape and sampled values.

Negative and malformed numeric CLI arguments are rejected.

The improved harness made it possible to separate:

1. First forward after mapping.
2. Warm forward passes on an established mapping.
3. Optional prefault work performed during loading.
4. Numerical reference checking performed outside the timed region.

---

## 7. Experiment 1: Thread Count Before Kernel Changes

The first sweep changed only the ability to configure thread count.

At sequence 200, the original-kernel warm median was approximately:

| Threads | Median latency |
|---:|---:|
| 12 | 441.4 ms |
| 16 | 402.3 ms |
| 20 | 352.6 ms |
| 30 | 429.7 ms |

This established two facts:

- Thirty threads were not a good default on the 20-logical-CPU host.
- Thread-count effects had to be controlled before attributing improvements to SIMD or mmap behavior.

At sequence 1, changing the requested thread count did not materially improve projection time, because the row-only decomposition still left the large mapped projections serialized.

---

## 8. Experiment 2: Explicit AVX2/FMA Micro-Kernel

### 8.1 Kernel design

A dedicated FP32 `6 × 16` micro-kernel was added for x86 hosts supporting AVX2 and FMA. Runtime dispatch preserves a scalar fallback.

The kernel uses 12 explicit YMM accumulators:

```text
6 rows × 2 vectors per row × 8 FP32 values per vector
```

Its inner loop:

1. Loads two eight-float vectors from B.
2. Broadcasts one scalar from each of six A rows.
3. Performs 12 vector FMA operations.
4. Repeats over the current K tile.
5. Stores C after the tile is complete.

The generated assembly was inspected after compilation. It showed:

- `vfmadd231ps` instructions in the inner loop.
- All 12 accumulators assigned to vector registers.
- No inner-loop accumulator spills to the stack.

This confirmed the central kernel hypothesis at the machine-code level rather than relying on source-code assumptions.

### 8.2 Alignment handling

Mapped tensor bases are page-aligned. For the real model shapes, weight row strides and 16-column microtile offsets also provide suitable wide-load alignment.

Two template instantiations were therefore generated:

- Verified aligned B loads.
- Safe unaligned B loads for arbitrary shapes.

C remains handled with unaligned loads and stores because ordinary scratch allocation does not guarantee equivalent alignment.

The implementation does not claim a separately measured, statistically significant speedup from alignment alone. Its primary validated benefit is that alignment assumptions are explicit and guarded rather than accidental.

### 8.3 K-tile experiment

A full-K variant was tested in which each C microtile stayed in accumulators across the entire reduction dimension.

The result was regime-dependent:

- It helped the small-M case.
- It substantially regressed sequence-200 execution by damaging cache behavior.

The final implementation therefore retains **64-wide K tiles**. This was a useful negative result: reducing C stores is not automatically beneficial if the loop-order change damages B locality.

---

## 9. Experiment 3: Better Work Decomposition

### 9.1 Six-row task units

For the row-oriented path, work is now expressed in units of the micro-kernel's six-row block rather than arbitrary row counts.

Benefits include:

- Fewer systematic one-row microtiles.
- More balanced complete microtiles across workers.
- Better B reuse per logical scan.
- A direct relationship between scheduling and kernel geometry.

### 9.2 Column partitioning for small M

When the number of available six-row blocks is smaller than the requested worker count, the mmap GEMM can partition N into 16-column tasks instead.

This is particularly important for `M = 1`:

- Workers own disjoint output columns.
- No split-K reduction is needed.
- A is small and read-only.
- Multiple cores can participate in streaming different weight panels.
- The mapped weights remain zero-copy.

The schedule is adaptive rather than assuming that either row-only or column-only partitioning is universally best.

### 9.3 Default thread count

The public execution paths now accept a runtime thread count. The default uses hardware concurrency instead of a hard-coded 30.

The benchmark still makes the count explicit for controlled experiments. Hardware concurrency is a reasonable default, not a claim that it is optimal on every hybrid CPU.

---

## 10. Experiment 4: In-Place SwiGLU Fusion

The MLP now computes:

```cpp
gate.silu_hadamard(up, threads, num_threads);
return gate.matmul(*W_down_, threads, num_threads);
```

Each element becomes:

```text
gate[i] = silu(gate[i]) * up[i]
```

This preserves the mathematical operation while eliminating:

- The separate `activated` matrix.
- The separate `fused` matrix.
- The standalone Hadamard output pass.
- Associated zero-initialization and first-touch traffic.

At sequence 200, this removes two approximately 8.4 MiB temporary matrices from the MLP pipeline.

During development, the median checkpoint improved from roughly 270 ms to 239 ms at 20 threads. These intermediate checkpoints were collected during sequential development and should be treated as indicative ablations, not as precise independent effect sizes.

---

## 11. Experiment 5: RoPE Reuse

A multi-head RoPE helper now precomputes sine and cosine values once per sequence position and dimension pair, then reuses them across heads.

Q and K are rotated before the per-query-head attention loop:

```cpp
apply_rope_heads(Q, n_heads, head_dim);
apply_rope_heads(K, n_kv_heads, head_dim);
```

This changes the work from repeated per-head transcendental evaluation to:

- One inverse-frequency computation per dimension pair.
- One sine/cosine table per position and pair.
- Reuse across all query heads.
- Exactly one rotation per unique KV head.

The transformation was validated against the original per-head implementation, including a nonzero position offset.

A development checkpoint at sequence 200 improved from approximately 239 ms to 229 ms, although the final controlled A/B result is the more reliable overall measurement.

---

## 12. Experiment 6: Mmap Prefaulting and Huge-Page Advice

### 12.1 Demand paging versus prefaulting

The loader accepts an optional `MMapLoadOptions` structure. With prefaulting enabled, `MAP_POPULATE` is added to the mapping flags.

This moves page-table population and file-page retrieval into mapping/load time. The benchmark reports load and forward times separately, preventing the work from simply being hidden outside the measurement.

For the one-token workload, five paired fresh-process runs produced approximate median totals:

| Policy | Load + first forward |
|---|---:|
| Demand paging | 226.9 ms |
| Prefault | 29.9 ms |

The first forward itself generally fell from the 215–233 ms range to approximately 12–19 ms, with roughly 12–21 ms spent in loading.

All these runs reported **zero major faults**. They therefore demonstrate a large benefit for first-touch execution with a warm/ambient OS page cache. They do not establish storage-cold performance.

### 12.2 Huge-page advice

An opt-in `MADV_HUGEPAGE` experiment was added and tested.

No useful improvement was observed. In addition, successful `madvise()` only means that advice was accepted; it does not prove that huge pages backed the mapping. The benchmark output was consequently named `hugepage_advice`, not simply `hugepages`.

No huge-page performance claim is made. Verifying actual backing would require inspecting information such as `/proc/self/smaps`, and eligibility depends on the kernel, filesystem, mapping permissions, and host configuration.

---

## 13. Experiment 7: Native Compiler Flags

A CMake option was added:

```bash
cmake -S . -B build-native \
  -DCMAKE_BUILD_TYPE=Release \
  -DTINYSERVE_NATIVE=ON
```

For the mmap benchmark, this adds `-march=native`.

After the explicit AVX2 kernel was introduced, native flags did not produce a consistent additional win:

- One sequence-200 comparison was approximately 211.4 ms generic versus 208.7 ms native.
- One sequence-1 comparison was approximately 28.5 ms generic versus 29.2 ms native.

The proper conclusion is not that native flags are useless, but that they are no longer a substitute for the explicit kernel and should be measured independently on each target machine.

---

## 14. Final Controlled Performance Results

The main comparison used preserved baseline and optimized executables with the same harness semantics.

Protocol:

- Identical converted weights.
- Identical seeded input.
- Twenty threads in both variants.
- Two warmup forwards.
- Five measured forwards per process.
- Five repeated processes per variant.
- Baseline and optimized runs interleaved to reduce drift.
- Warm mappings, not storage-cold cache eviction.

The table reports the median of the five process-level medians:

| Sequence | Baseline | Optimized | Speedup | Latency reduction |
|---:|---:|---:|---:|---:|
| 200 | 340.4 ms | 205.5 ms | 1.66× | 39.6% |
| 1 | 67.4 ms | 28.3 ms | 2.38× | 58.0% |

The optimized sequence-200 thread sweep also showed:

| Threads | Median latency |
|---:|---:|
| 8 | 214.3 ms |
| 12 | 184.9 ms |
| 16 | 194.0 ms |
| 20 | 207.8 ms |
| 24 | 284.4 ms |
| 30 | 390.0 ms |

The best measured sequence-200 configuration in that sweep was therefore 12 threads, not the original 30.

For sequence 1, results were broadly in the 26–34 ms range across many thread counts, consistent with a workload strongly influenced by weight bandwidth and first-touch behavior rather than compute scaling alone.

---

## 15. Correctness and Safety Validation

### 15.1 Independent GEMM reference

A new mmap matmul test compares results against an independent naive triple-loop reference with double-precision accumulation.

Tested shapes include:

```text
(1, 1, 1)
(5, 7, 15)
(6, 64, 16)
(7, 65, 17)
(65, 130, 70)
```

These exercise:

- Full microtiles.
- M, N, and K tails.
- Cache-tile boundaries.
- Row-oriented and column-oriented schedules.
- Multiple thread counts.

The test does not rely on the ordinary GEMM as its only reference, avoiding a situation where the same kernel bug appears on both sides.

### 15.2 Generic scalar path

A dedicated test verifies that `Mat<double>` can consume FP32 mapped weights through the scalar path. This prevents the AVX2 specialization from accidentally making the generic API float-only.

### 15.3 SwiGLU and RoPE tests

New tests validate:

- In-place `silu(gate) * up` against the explicit formula.
- Multi-head RoPE against the original per-head operation.
- Nonzero position offsets.
- Multiple heads and nontrivial dimensions.

### 15.4 Full-block differential check

The benchmark's `--check` option copies the mapped FP32 values into the ordinary block, executes both implementations on the same input, and compares every output element outside the timed region.

For the real 200-token fixture, the result was:

```text
max_abs_error = 1.66893005e-06
status = ok
```

Small differences are expected because FMA changes rounding behavior. The comparison uses explicit absolute/relative tolerances and rejects non-finite values in either implementation.

### 15.5 Test and sanitizer results

The following passed:

- All four existing CTest suites.
- The new independent mmap matmul tests.
- The new generic-type, SwiGLU, and RoPE tests.
- AddressSanitizer.
- UndefinedBehaviorSanitizer.
- A real mmap block forward under the sanitizer build.
- Warning-enabled benchmark compilation with `-Wall -Wextra -Wpedantic`.
- `git diff --check`.

---

## 16. Files Changed

| File | Purpose |
|---|---|
| `include/mat.h` | AVX2/FMA kernel, aligned/unaligned paths, adaptive mmap scheduling, runtime thread counts, in-place SwiGLU primitive |
| `include/llama_mmap_block.h` | Thread-count propagation, shared multi-head RoPE, single K rotation, fused MLP execution |
| `include/mmap_block_loader.h` | Optional prefault and huge-page-advice policies with RAII-preserving error handling |
| `bench_block_mmap.cpp` | Repeatable benchmark CLI, statistics, fault counters, prefault experiments, full differential reference check |
| `tests/test_mat.cpp` | Independent mmap GEMM reference tests, generic-type test, in-place SwiGLU test |
| `tests/test_gqa.cpp` | Multi-head mmap RoPE equivalence test |
| `CMakeLists.txt` | Warning flags and optional native-ISA benchmark build |

---

## 17. Reproducing the Experiments

### 17.1 Release build and tests

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

### 17.2 Sequence-200 warm benchmark with full reference check

```bash
./build/bench_block_mmap bloc_0 \
  --seq 200 \
  --threads 12 \
  --warmup 2 \
  --iters 10 \
  --check
```

### 17.3 Sequence-1 warm benchmark

```bash
./build/bench_block_mmap bloc_0 \
  --seq 1 \
  --threads 16 \
  --warmup 2 \
  --iters 10
```

### 17.4 First-touch comparison

Demand paging:

```bash
./build/bench_block_mmap bloc_0 \
  --seq 1 \
  --threads 20 \
  --warmup 0 \
  --iters 1
```

Prefaulting:

```bash
./build/bench_block_mmap bloc_0 \
  --seq 1 \
  --threads 20 \
  --warmup 0 \
  --iters 1 \
  --prefault
```

For a fair first-touch comparison, include both `load_ms` and the first forward time.

### 17.5 Thread sweep

```bash
for threads in 4 8 12 16 20 24 30; do
  ./build/bench_block_mmap bloc_0 \
    --seq 200 \
    --threads "$threads" \
    --warmup 2 \
    --iters 5
done
```

### 17.6 Native-ISA build

```bash
cmake -S . -B build-native \
  -DCMAKE_BUILD_TYPE=Release \
  -DTINYSERVE_NATIVE=ON
cmake --build build-native --target bench_block_mmap -j2
```

### 17.7 Sanitizer build

```bash
cmake -S . -B build-sanitize \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS='-Wall -Wextra -Wpedantic -fsanitize=address,undefined -fno-omit-frame-pointer' \
  -DCMAKE_EXE_LINKER_FLAGS='-fsanitize=address,undefined'

cmake --build build-sanitize -j2
ASAN_OPTIONS=detect_leaks=1 \
  ctest --test-dir build-sanitize --output-on-failure
```

---

## 18. Limitations

The results should be interpreted with the following constraints:

1. **Single host.** Measurements were collected on one hybrid Intel laptop CPU. Optimal thread counts will differ elsewhere.
2. **Performance noise.** Hybrid scheduling, frequency changes, and background activity caused visible run-to-run variation. Repeated and interleaved measurements were therefore essential.
3. **No storage-cold claim.** The prefault experiments reported zero major faults and did not use privileged system-wide cache eviction.
4. **Decode-like shape only.** Sequence 1 does not represent a complete production KV-cached decoding pipeline.
5. **One transformer block.** Results are not automatically full-model throughput numbers.
6. **FP32 only for the optimized kernel.** Quantized mmap inference remains unimplemented.
7. **Huge-page usage not verified.** Only mapping advice was tested; actual huge-page backing was not established.
8. **Synthetic input.** The benchmark uses deterministic random input rather than real embedding trajectories.
9. **No persistent worker pool.** Threads are still recreated at operation boundaries, although decomposition and oversubscription were improved.

---

## 19. Recommended Next Experiments

### 19.1 Persistent thread pool

This is the clearest remaining scheduling experiment. Preserve the current arithmetic and task decomposition while replacing repeated create/join episodes with a reusable pool.

Measure:

- Thread creation and synchronization time.
- Small-matrix and per-head attention latency.
- Behavior with concurrent requests.
- Spin-versus-block waiting policies.

A pool should not introduce one oversubscribed team per concurrently running request.

### 19.2 Packed weight panels

The current mmap matrices remain in ordinary row-major form. A persistent prepacked layout could match the kernel's K/N traversal, improve TLB behavior, and simplify aligned loads.

Packing cost must be accounted for honestly:

- Runtime packing damages zero-copy and first-use latency.
- Offline packing can preserve a read-only mapped deployment model.
- Repeated inference can amortize initialization; single-use cold inference may not.

### 19.3 Quantized mmap weights

For true small-M decoding, unique weight traffic is a first-order constraint. INT8, INT4, or BF16/FP16 mapped formats could reduce bytes read, page faults, and TLB pressure.

A useful prototype should measure:

- Cold bytes read.
- Warm memory bandwidth.
- Dequantization instructions.
- Numerical error.
- Total load-plus-first-token and steady-state token latency.

Runtime dequantization into full FP32 buffers should be treated as a different design from zero-copy quantized mmap.

### 19.4 Scratch-buffer reuse and additional fusion

An execution-local arena could reduce repeated zero-initialization and anonymous page faults. Further candidates include:

- Residual additions fused into projection epilogues.
- Direct strided head views instead of repeated Q/K/V slices.
- Reused K transposes and attention score buffers.
- Direct output writes into the concatenated attention buffer.

Shared scratch state must not make concurrent forwards unsafe.

### 19.5 Causal attention work elimination

The current attention computes full square score matrices before masking. Triangular evaluation can avoid approximately half of QK and attention-value arithmetic at large sequence lengths, as well as exponentials for masked entries.

This is lower priority than projections at sequence 200 because attention GEMMs are only about 1% of total GEMM FLOPs at that shape, but its importance grows with sequence length.

### 19.6 Rigorous cold-I/O protocol

A future cold-start study should distinguish:

1. Storage-cold and page-cache-cold.
2. Page-cache-warm but newly mapped.
3. Established mapping and steady-state execution.

It should record actual storage reads, residency, major/minor faults, and total initialization-plus-first-output latency. Dropping system caches should only be done on an isolated host with explicit permission.

---

## 20. Conclusions

The largest gains did not come from changing `mmap()` flags. They came from fixing the interaction between kernel geometry, parallel decomposition, and workload shape.

The most important accepted changes were:

- A real register-resident AVX2/FMA FP32 micro-kernel.
- Retention of cache-friendly 64-wide K tiles.
- Six-row task partitioning for large M.
- Column partitioning for small M.
- Reduced thread oversubscription.
- In-place SwiGLU fusion.
- Shared RoPE trigonometry and single rotation of each K head.
- Optional prefaulting for first-touch latency.
- A benchmark capable of distinguishing these effects and checking numerical correctness.

The resulting implementation is approximately **1.66× faster on the controlled 200-token workload** and **2.38× faster on the controlled one-token workload**, with the largest first-touch improvement coming from prefaulting.

Equally importantly, the negative experiments established boundaries for future work: full-K accumulation was not universally beneficial, native compiler flags were not a substitute for a good kernel, and huge-page advice did not provide a demonstrated gain. The next highest-value investigations are persistent workers, packed or quantized mapped weight layouts, and a true KV-cached decoding benchmark.
