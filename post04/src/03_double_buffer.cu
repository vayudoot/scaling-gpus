// 03_double_buffer.cu — Post 4: CUDA Streams and Async Execution
//
// The double-buffer pipeline: the canonical pattern for keeping both the
// DMA copy engine and the compute SMs busy simultaneously.
//
// Single-buffer (naive) timeline:
//   [Copy 0][Kernel 0][Copy 1][Kernel 1][Copy 2][Kernel 2]...
//   Copy engine idle during every kernel.
//   Compute engine idle during every copy.
//
// Double-buffer pipeline timeline:
//   [Copy 0][Kernel 0]
//           [Copy 1][Kernel 1]        ← copy 1 overlaps kernel 0
//                   [Copy 2][Kernel 2] ← copy 2 overlaps kernel 1
//   After the first copy primes the pipeline, every subsequent step
//   overlaps copy N+1 with kernel N.
//
// Required ingredients:
//   1. TWO device buffers (ping-pong between them)
//   2. TWO host buffers, both PINNED (cudaMallocHost)
//   3. TWO streams (one per buffer slot)
//   4. cudaMemcpyAsync for all copies
//
// How much speedup to expect:
//   If copy_time ≈ kernel_time → approaches 2×
//   If one dominates the other, speedup is smaller but still present.
//   Measure both individually first (see the baseline section below).
//
// NVTX annotations are included so the timeline in Nsight Systems shows
// clearly labelled regions for each phase.

#include <cuda_runtime.h>
// NVTX — header-only, no linking needed.
// CUDA 12+: nvtx3/nvToolsExt.h   CUDA 11: nvToolsExt.h
// We try both; if neither is available the push/pop calls become no-ops.
#if __has_include(<nvtx3/nvToolsExt.h>)
  #include <nvtx3/nvToolsExt.h>
#elif __has_include(<nvToolsExt.h>)
  #include <nvToolsExt.h>
#else
  // No NVTX available — define stubs so the code compiles
  typedef struct { int version; int size; unsigned colorType; unsigned color;
                   unsigned messageType; union { const char* ascii; } message; }
      nvtxEventAttributes_t;
  #define NVTX_VERSION 2
  #define NVTX_EVENT_ATTRIB_STRUCT_SIZE sizeof(nvtxEventAttributes_t)
  #define NVTX_COLOR_ARGB 1
  #define NVTX_MESSAGE_TYPE_ASCII 1
  inline void nvtxRangePushEx(const nvtxEventAttributes_t*) {}
  inline void nvtxRangePop() {}
  #warning "NVTX headers not found — timeline annotations disabled"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../include/timer.cuh"

// ─────────────────────────────────────────────────────────────────────────────
// Process kernel: scale every element and accumulate a checksum
// ─────────────────────────────────────────────────────────────────────────────

__global__ void processKernel(const float* __restrict__ in,
                               float*       __restrict__ out,
                               float scalar, int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = in[i];
    for (int it = 0; it < iters; it++)
        v = v * scalar + 0.0001f;
    out[i] = v;
}

// ─────────────────────────────────────────────────────────────────────────────
// Simulate loading the next batch on the host (e.g. reading from disk/network)
// In a real system this would be your DataLoader.
// ─────────────────────────────────────────────────────────────────────────────

static void fillHostBatch(float* buf, int batch_id, int n) {
    float base = (float)batch_id * 0.01f;
    for (int i = 0; i < n; i++) buf[i] = base + (float)i * 0.000001f;
}

// ─────────────────────────────────────────────────────────────────────────────
// NVTX colour codes for the Nsight Systems timeline
// ─────────────────────────────────────────────────────────────────────────────

static void nvtx_push(const char* name, unsigned colour) {
    nvtxEventAttributes_t attr = {};
    attr.version       = NVTX_VERSION;
    attr.size          = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attr.colorType     = NVTX_COLOR_ARGB;
    attr.color         = colour;
    attr.messageType   = NVTX_MESSAGE_TYPE_ASCII;
    attr.message.ascii = name;
    nvtxRangePushEx(&attr);
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    print_device_info();

    const int NUM_BATCHES  = (argc > 1) ? atoi(argv[1]) : 20;
    const int BATCH_ELEMS  = 1 << 22;   // 4 M floats = 16 MB per batch
    const int KERNEL_ITERS = 80;         // tune to make copy ≈ kernel time
    const int BLOCK        = 256;
    const int GRID         = (BATCH_ELEMS + BLOCK - 1) / BLOCK;
    size_t    batch_bytes  = (size_t)BATCH_ELEMS * sizeof(float);

    printf("Batches       : %d\n", NUM_BATCHES);
    printf("Batch size    : %d floats (%.0f MB)\n",
           BATCH_ELEMS, (double)batch_bytes / 1e6);
    printf("Kernel iters  : %d\n\n", KERNEL_ITERS);

    // ── Allocate: two pinned host buffers + two device buffers ───────────────
    float* h_buf[2];
    float* d_in[2];
    float* d_out[2];

    for (int s = 0; s < 2; s++) {
        CUDA_CHECK(cudaMallocHost(&h_buf[s], batch_bytes));   // PINNED
        CUDA_CHECK(cudaMalloc(&d_in[s],  batch_bytes));
        CUDA_CHECK(cudaMalloc(&d_out[s], batch_bytes));
    }

    // ── Two streams ──────────────────────────────────────────────────────────
    cudaStream_t stream[2];
    for (int s = 0; s < 2; s++)
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream[s], cudaStreamNonBlocking));

    // ── Baseline: individual copy and kernel times ────────────────────────────
    section("Baseline timings");
    {
        fillHostBatch(h_buf[0], 0, BATCH_ELEMS);
        CUDA_CHECK(cudaMemcpy(d_in[0], h_buf[0], batch_bytes, cudaMemcpyHostToDevice));

        GpuTimer t;
        t.start();
        CUDA_CHECK(cudaMemcpy(d_in[0], h_buf[0], batch_bytes, cudaMemcpyHostToDevice));
        float ms_copy = t.stop_ms();

        t.start();
        processKernel<<<GRID, BLOCK>>>(d_in[0], d_out[0], 1.0001f, BATCH_ELEMS, KERNEL_ITERS);
        float ms_kernel = t.stop_ms();

        printf("  H→D copy  : %.2f ms  (%.1f GB/s)\n",
               ms_copy, bandwidth_gb_s(batch_bytes, ms_copy));
        printf("  Kernel    : %.2f ms\n", ms_kernel);
        printf("  If fully overlapping, pipeline step ≈ max(%.2f, %.2f) = %.2f ms\n\n",
               ms_copy, ms_kernel, fmaxf(ms_copy, ms_kernel));
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Naive (single-buffer, single-stream)
    // ══════════════════════════════════════════════════════════════════════════
    section("Naive: serial copy then compute, one buffer");
    {
        fillHostBatch(h_buf[0], 0, BATCH_ELEMS);
        CUDA_CHECK(cudaMemcpy(d_in[0], h_buf[0], batch_bytes, cudaMemcpyHostToDevice));
        processKernel<<<GRID, BLOCK>>>(d_in[0], d_out[0], 1.f, BATCH_ELEMS, KERNEL_ITERS);
        CUDA_CHECK(cudaDeviceSynchronize());  // warm up

        double t0 = wall_ms();
        nvtx_push("naive", 0xFF607060);

        for (int b = 0; b < NUM_BATCHES; b++) {
            char label[32]; snprintf(label, sizeof(label), "copy_%d", b);
            nvtx_push(label, 0xFF4080C0);
            fillHostBatch(h_buf[0], b, BATCH_ELEMS);
            CUDA_CHECK(cudaMemcpy(d_in[0], h_buf[0], batch_bytes,
                                  cudaMemcpyHostToDevice));    // BLOCKS CPU
            nvtxRangePop();

            snprintf(label, sizeof(label), "kernel_%d", b);
            nvtx_push(label, 0xFF40A060);
            processKernel<<<GRID, BLOCK>>>(d_in[0], d_out[0],
                                           1.0001f, BATCH_ELEMS, KERNEL_ITERS);
            CUDA_CHECK(cudaDeviceSynchronize());
            nvtxRangePop();
        }
        nvtxRangePop();
        double ms_naive = wall_ms() - t0;
        printf("  Total wall time : %.2f ms  (%.2f ms/batch)\n",
               ms_naive, ms_naive / NUM_BATCHES);
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Double-buffer pipeline
    // ══════════════════════════════════════════════════════════════════════════
    section("Double-buffer pipeline: copy and compute overlap");
    {
        // ── Warm up ──────────────────────────────────────────────────────────
        fillHostBatch(h_buf[0], 0, BATCH_ELEMS);
        CUDA_CHECK(cudaMemcpyAsync(d_in[0], h_buf[0], batch_bytes,
                                   cudaMemcpyHostToDevice, stream[0]));
        processKernel<<<GRID, BLOCK, 0, stream[0]>>>(d_in[0], d_out[0],
                                                       1.f, BATCH_ELEMS, 1);
        CUDA_CHECK(cudaDeviceSynchronize());

        double t0 = wall_ms();
        nvtx_push("pipeline", 0xFF806030);

        // ── Pipeline primer: copy batch 0 ────────────────────────────────────
        // Before the main loop starts, copy the very first batch so the kernel
        // on step 0 has data to process immediately.
        int cur  = 0;
        fillHostBatch(h_buf[cur], 0, BATCH_ELEMS);
        CUDA_CHECK(cudaMemcpyAsync(d_in[cur], h_buf[cur], batch_bytes,
                                   cudaMemcpyHostToDevice, stream[cur]));

        for (int b = 0; b < NUM_BATCHES; b++) {
            cur        = b % 2;
            int nxt    = 1 - cur;

            // Wait for this slot's copy to finish before we compute on it.
            // (The copy was issued on stream[cur] in the previous iteration,
            //  or above for b=0.  Compute must not start before copy lands.)
            char label[48];
            snprintf(label, sizeof(label), "sync_copy_%d", b);
            nvtx_push(label, 0xFF706060);
            CUDA_CHECK(cudaStreamSynchronize(stream[cur]));
            nvtxRangePop();

            // Launch compute on current slot — occupies stream[cur] SMs
            snprintf(label, sizeof(label), "kernel_%d", b);
            nvtx_push(label, 0xFF40A060);
            processKernel<<<GRID, BLOCK, 0, stream[cur]>>>(
                d_in[cur], d_out[cur], 1.0001f, BATCH_ELEMS, KERNEL_ITERS);
            nvtxRangePop();

            // Simultaneously: load next batch into the other slot
            if (b + 1 < NUM_BATCHES) {
                snprintf(label, sizeof(label), "copy_%d", b + 1);
                nvtx_push(label, 0xFF4080C0);
                fillHostBatch(h_buf[nxt], b + 1, BATCH_ELEMS);
                CUDA_CHECK(cudaMemcpyAsync(d_in[nxt], h_buf[nxt], batch_bytes,
                                           cudaMemcpyHostToDevice, stream[nxt]));
                nvtxRangePop();
                // At this point:
                //   stream[cur]: running processKernel on batch b
                //   stream[nxt]: DMA-copying batch b+1
                //   Both are happening in parallel on separate hardware units.
            }
        }

        // Drain: wait for the last kernel to finish
        CUDA_CHECK(cudaStreamSynchronize(stream[cur]));
        nvtxRangePop();  // pipeline

        double ms_pipe = wall_ms() - t0;
        printf("  Total wall time : %.2f ms  (%.2f ms/batch)\n",
               ms_pipe, ms_pipe / NUM_BATCHES);
    }

    // ── Cleanup ───────────────────────────────────────────────────────────────
    for (int s = 0; s < 2; s++) {
        CUDA_CHECK(cudaStreamDestroy(stream[s]));
        CUDA_CHECK(cudaFreeHost(h_buf[s]));
        cudaFree(d_in[s]); cudaFree(d_out[s]);
    }

    printf("\n");
    printf("Profile with Nsight Systems to see the overlap:\n");
    printf("  nsys profile --trace=cuda,nvtx ./build/03_double_buffer\n");
    printf("  NVTX labels show each copy and kernel phase.\n");
    printf("  In the pipeline section, copy and kernel bars should overlap.\n\n");
    printf("Tuning tips:\n");
    printf("  - Increase KERNEL_ITERS if the kernel is much faster than the copy\n");
    printf("  - Best overlap when copy_time ≈ kernel_time\n");
    printf("  - If copy >> kernel: consider larger batches\n");
    printf("  - If kernel >> copy: consider triple-buffering (3 slots)\n");

    return 0;
}
