# Scaling GPUs — Post 4: CUDA Streams and Async Execution

Code for **Post 4** of the [Scaling GPUs](https://substack.com/TODO) series.
This post covers the patterns that keep a GPU genuinely busy instead of
alternating between idle and active.

---

## What's in here

```
scaling_gpus_post04/
├── include/
│   └── timer.cuh              GpuTimer, CUDA_CHECK, wall_ms, section()
├── src/
│   ├── 01_default_stream.cu   Stream 0 serialisation vs non-default streams
│   ├── 02_async_copy.cu       cudaMemcpyAsync: pageable (silent sync) vs pinned
│   ├── 03_double_buffer.cu    Double-buffer pipeline with NVTX annotations
│   ├── 04_events.cu           Events for timing and cross-stream dependency
│   └── 05_cuda_graphs.cu      CUDA Graphs: capture once, replay many times
├── scripts/
│   └── run_all.sh
└── Makefile
```

---

## Quick start

```bash
cd scaling_gpus_post04
bash scripts/run_all.sh           # auto-detect GPU, build, run all
# or:
make && ./build/03_double_buffer 20
```

**NVTX note:** Program 3 uses NVTX for timeline annotations.
CUDA 12 includes NVTX3 as a header-only library at
`$(CUDA_HOME)/include/nvtx3/`.  No extra install needed.
If your CUDA version is older, install NVTX separately or remove the
`#include <nvtx3/nvToolsExt.h>` lines and the `nvtx_push/pop` calls —
everything still runs, just without the named timeline regions.

---

## Programs

### 1. `01_default_stream` — The synchronising stream

The default stream (stream 0) is special: it waits for every previously
issued operation before it starts, and every subsequent operation waits for
it. Two kernels on the default stream are always serial, even if they are
completely independent.

Three experiments:
- Individual kernel times (reference)
- Both kernels on the default stream → serial, total ≈ sum
- Both kernels on non-default streams → concurrent, total ≈ max

```bash
./build/01_default_stream
```

**Profile to see the timeline:**
```bash
make profile-streams
# Open profiles/streams.nsys-rep in Nsight Systems GUI
# Compare the 'CUDA HW' rows: back-to-back vs overlapping bars
```

**Key rule:** never use the default stream (stream 0) in production kernels.
Always pass an explicit `cudaStream_t` to every launch and `cudaMemcpy*`.

---

### 2. `02_async_copy` — The pinned memory requirement

`cudaMemcpyAsync` is not actually asynchronous unless the host pointer comes
from `cudaMallocHost` (pinned / page-locked memory).

With regular `malloc` memory:
- The API returns without error
- Internally, CUDA stages through a pinned bounce buffer
- The GPU timeline is still sequential — no overlap occurs

With `cudaMallocHost` memory:
- The DMA engine directly reads from the stable physical address
- No CPU involvement after the transfer starts
- Genuine compute-copy overlap is possible

```bash
./build/02_async_copy
```

Expected output: pageable async copy gives similar wall time to synchronous
copy (Case A ≈ Case B). Pinned async copy (Case C) is faster because copy
and kernel run in parallel.

**The one-line fix:**
```cuda
// Before (no overlap possible):
float* h = (float*)malloc(bytes);

// After (genuine async overlap):
float* h;
cudaMallocHost(&h, bytes);    // page-locked
// ... use h ...
cudaFreeHost(h);              // matching free
```

---

### 3. `03_double_buffer` — The production pipeline pattern

The double-buffer pipeline keeps both the DMA copy engine and the SM compute
engine busy simultaneously. After the first batch primes the pipeline, every
subsequent step overlaps:
- Copy of batch N+1 happening on one stream
- Compute on batch N happening on the other stream

```bash
./build/03_double_buffer          # 20 batches (default)
./build/03_double_buffer 50       # more batches for cleaner timing
```

**Requirements for overlap:**
1. Two device buffers (ping and pong)
2. Two **pinned** host buffers (`cudaMallocHost`)
3. Two non-default streams
4. `cudaMemcpyAsync` for all copies

**Profile with NVTX labels:**
```bash
make profile-pipeline
```
The Nsight Systems timeline shows colour-coded `copy_N` and `kernel_N` regions.
In the pipeline section, these should interleave rather than appear sequential.

**How much speedup to expect:**
- If copy_time ≈ kernel_time → approaches 2×
- If one dominates → smaller speedup, but always non-negative
- Tune `KERNEL_ITERS` in the source to balance them

**The implementation pattern (simplified):**
```cuda
// Prime: copy batch 0 before the loop
cudaMemcpyAsync(d_in[0], h_buf[0], bytes, H2D, stream[0]);

for (int b = 0; b < N; b++) {
    int cur = b % 2, nxt = 1 - cur;

    // Wait for this slot's copy
    cudaStreamSynchronize(stream[cur]);

    // Compute on current slot
    kernel<<<grid, block, 0, stream[cur]>>>(d_in[cur], d_out[cur]);

    // Copy next batch into the other slot — runs concurrently with kernel
    if (b + 1 < N) {
        fill(h_buf[nxt], b + 1);
        cudaMemcpyAsync(d_in[nxt], h_buf[nxt], bytes, H2D, stream[nxt]);
    }
}
cudaStreamSynchronize(stream[cur]); // drain
```

---

### 4. `04_events` — Precision timing and cross-stream dependencies

CUDA events serve two roles:

**Timing:**
```cuda
cudaEvent_t start, stop;
cudaEventCreate(&start); cudaEventCreate(&stop);
cudaEventRecord(start);
myKernel<<<grid, block>>>();
cudaEventRecord(stop);
cudaEventSynchronize(stop);       // CPU waits here
float ms;
cudaEventElapsedTime(&ms, start, stop);
```
Accuracy: ±0.5 µs. Never use CPU timers (`clock()`, `gettimeofday()`) for
GPU kernels — kernel launches are asynchronous and the CPU returns before
the kernel runs.

**Cross-stream dependency (GPU-side, CPU-free):**
```cuda
kernelA<<<grid, block, 0, streamA>>>(d_buf);
cudaEventRecord(event_A_done, streamA);   // GPU writes timestamp

// streamB will pause at this fence until event_A_done fires.
// The CPU continues immediately — it does NOT block.
cudaStreamWaitEvent(streamB, event_A_done, 0);
kernelB<<<grid, block, 0, streamB>>>(d_buf);  // uses kernelA's output
```

Contrast with `cudaStreamSynchronize(streamA)` which would block the **CPU**,
preventing it from submitting work to streamB or any other stream.

```bash
./build/04_events
```

---

### 5. `05_cuda_graphs` — Eliminating per-launch overhead

Every kernel launch costs roughly 5–20 µs of CPU overhead. For kernels that
run for hundreds of milliseconds this is negligible. For kernels that run in
microseconds — common in inference and small model steps — it dominates.

CUDA Graphs eliminate this by recording a sequence of launches once and
replaying the entire sequence as a single GPU command.

```bash
./build/05_cuda_graphs
```

**Three sections:**
1. **Tiny kernels:** launch overhead is a large fraction of kernel time.
   Graph replay shows dramatic speedup.
2. **Multi-kernel step:** 8 kernels per step, larger kernels.
   Speedup smaller but present.
3. **Updating parameters:** how to change scalar args and pointer targets
   between replays without re-capturing the graph.

**The capture / replay pattern:**
```cuda
// ── Capture (once, outside the hot loop) ─────────────────────────────
cudaGraph_t graph;
cudaGraphExec_t instance;

cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
// ... all kernel launches and memsets you want to capture ...
kernelA<<<grid, block, 0, stream>>>(args...);
kernelB<<<grid, block, 0, stream>>>(args...);
cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);

// ── Replay (many times in the hot loop) ──────────────────────────────
for (int step = 0; step < NUM_STEPS; step++) {
    // Optional: update d_input contents here (same pointer, new data)
    cudaGraphLaunch(instance, stream);
    cudaStreamSynchronize(stream);  // or use an event if batching replays
}

// Cleanup
cudaGraphExecDestroy(instance);
cudaGraphDestroy(graph);
```

**When graphs don't help:**
- Dynamic shapes (tensor sizes change per step)
- Conditional logic that changes which kernels run
- Large kernels where 20 µs overhead is a rounding error

**Profile to see the gap difference:**
```bash
make profile-graphs
# Compare the inter-kernel gaps in 'eager' vs 'graph' sections of the timeline
```

---

## The mental model

```
Default stream:  op1 ──── op2 ──── op3        (serial, each waits for the last)
Non-default:     stream0: op1 ──── op3
                 stream1:      op2             (concurrent where hardware permits)
                 dependency:   event fence (GPU-side, CPU-free)

Async copy:      stream0: [H→D copy N+1] ──── [H→D copy N+2]
                 stream1:                [kernel N] ──── [kernel N+1]
                          ↑ overlap here (pinned memory required)

CUDA Graph:      capture: record op1, op2, op3 once
                 replay:  submit all three as one GPU command (near-zero CPU cost)
```

---

## Profiling cheat sheet

```bash
# Full timeline — see stream occupancy
nsys profile --trace=cuda,nvtx ./build/03_double_buffer 30

# See kernel-to-kernel gaps (launch overhead)
nsys profile --trace=cuda ./build/05_cuda_graphs

# Nsight Compute — is the GPU waiting on the CPU?
ncu --metrics sm__cycles_elapsed.avg.per_second ./build/05_cuda_graphs

# Check: are any kernels running concurrently?
# In Nsight Systems, look at the 'CUDA HW → Kernels' row.
# Overlapping bars = concurrent. Back-to-back = serial.
```

---

## Exercises

**Exercise 1 (streams):** Modify `01_default_stream.cu` to run four kernels
concurrently on four separate streams.  Measure the wall time and compare to
running them all on the default stream.  At what kernel count does the GPU
run out of SMs to schedule them all simultaneously?

**Exercise 2 (async copy):** In `02_async_copy.cu`, add a D→H copy (device
to host) and overlap it with a kernel running on a separate stream.  Both the
D→H copy and the kernel should run concurrently.  Verify with Nsight Systems.

**Exercise 3 (pipeline):** Modify `03_double_buffer.cu` into a **triple-buffer**
pipeline (three slots, three streams).  When does triple-buffering help over
double-buffering?  (Hint: when the kernel time is significantly longer than
the copy time.)

**Exercise 4 (events):** Rewrite the pipeline in `03_double_buffer.cu` to use
events instead of `cudaStreamSynchronize` for the slot-ready signal.  The CPU
should never block between batches.  Compare wall time to the
`cudaStreamSynchronize` version.

**Exercise 5 (graphs):** Capture a graph that includes a `cudaMemsetAsync`
in addition to two kernels.  Update the graph's kernel pointer to point to a
different device buffer between replays.  Verify that results are correct.

---

## What's next

**Post 5 — Building a Neural Net on One GPU**
The double-buffer pipeline from this post is exactly what production PyTorch
data loaders use (with `pin_memory=True` and prefetch workers).  Post 5 builds
the model side: forward pass, backpropagation, and a training loop using the
kernel patterns from Posts 2–3 and the async execution from Post 4.

---

## License

MIT.
