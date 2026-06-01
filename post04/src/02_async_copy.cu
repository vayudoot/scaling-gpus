// 02_async_copy.cu — Post 4: CUDA Streams and Async Execution
//
// Demonstrates cudaMemcpyAsync and the requirements for genuine overlap
// between a host-to-device copy and a compute kernel running on a different
// stream.
//
// Three experiments:
//   A. Synchronous copy + compute: fully serial, GPU idle during copies
//   B. Async copy with pageable host memory: silently falls back to sync
//      (no error, no warning, no overlap)
//   C. Async copy with PINNED host memory: genuine overlap
//
// The most important lesson: cudaMemcpyAsync on pageable (malloc) memory
// is NOT asynchronous.  The DMA engine requires a stable physical address,
// which pageable memory cannot guarantee because the OS can move it.
// Only cudaMallocHost (pinned / page-locked) memory enables true overlap.
//
// What to observe in Nsight Systems:
//   - Case A: copy bar → gap → kernel bar (sequential)
//   - Case B: looks async on the CPU, but GPU timeline still sequential
//   - Case C: copy bar and kernel bar overlapping in the GPU timeline

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/timer.cuh"

// ─────────────────────────────────────────────────────────────────────────────
// Kernel: process a buffer that was already copied to the device.
// Intentionally heavier than the copy to make relative timing clear.
// ─────────────────────────────────────────────────────────────────────────────

__global__ void processKernel(float* data, float scalar, int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = data[i];
    for (int it = 0; it < iters; it++)
        v = v * scalar + (1.0f - scalar);
    data[i] = v;
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    print_device_info();

    const int N        = 1 << 24;   // 16 M floats = 64 MB
    const int ITERS    = 100;        // kernel work per element
    const int BLOCK    = 256;
    const int GRID     = (N + BLOCK - 1) / BLOCK;
    const int REPS     = 5;
    size_t    bytes    = (size_t)N * sizeof(float);

    // ── Host memory: two flavours ─────────────────────────────────────────────
    float* h_pageable = (float*)malloc(bytes);              // standard heap
    float* h_pinned;
    CUDA_CHECK(cudaMallocHost(&h_pinned, bytes));            // page-locked

    for (int i = 0; i < N; i++) {
        h_pageable[i] = (float)i * 0.001f;
        h_pinned[i]   = (float)i * 0.001f;
    }

    // Device buffers: one for the transfer target, one already on device
    // (so the kernel can run while the copy fills the other)
    float *d_buf, *d_compute;
    CUDA_CHECK(cudaMalloc(&d_buf,     bytes));
    CUDA_CHECK(cudaMalloc(&d_compute, bytes));
    CUDA_CHECK(cudaMemset(d_compute, 1, bytes));

    // ── Measure raw copy bandwidth ────────────────────────────────────────────
    section("Baseline: raw H→D copy bandwidth");
    {
        // Pageable
        CUDA_CHECK(cudaMemcpy(d_buf, h_pageable, bytes, cudaMemcpyHostToDevice));
        GpuTimer t; t.start();
        for (int r = 0; r < REPS; r++)
            CUDA_CHECK(cudaMemcpy(d_buf, h_pageable, bytes, cudaMemcpyHostToDevice));
        float ms_page = t.stop_ms() / REPS;

        // Pinned
        CUDA_CHECK(cudaMemcpy(d_buf, h_pinned, bytes, cudaMemcpyHostToDevice));
        t.start();
        for (int r = 0; r < REPS; r++)
            CUDA_CHECK(cudaMemcpy(d_buf, h_pinned, bytes, cudaMemcpyHostToDevice));
        float ms_pin = t.stop_ms() / REPS;

        printf("  Pageable H→D : %.2f ms  (%.1f GB/s)\n",
               ms_page, bandwidth_gb_s(bytes, ms_page));
        printf("  Pinned   H→D : %.2f ms  (%.1f GB/s)\n",
               ms_pin,  bandwidth_gb_s(bytes, ms_pin));
        printf("  (Pinned is typically 2–3× faster: no kernel-mode copy, no page faults)\n");
    }

    // ── Case A: synchronous copy, then compute ────────────────────────────────
    section("Case A: synchronous cudaMemcpy + compute (no overlap possible)");
    {
        // Warm up kernel
        processKernel<<<GRID, BLOCK>>>(d_compute, 1.0001f, N, ITERS);
        CUDA_CHECK(cudaDeviceSynchronize());

        double t0 = wall_ms();
        for (int r = 0; r < REPS; r++) {
            // Sync copy blocks the CPU (and therefore the CUDA command queue)
            // until the copy is done. Kernel launch cannot be submitted until then.
            CUDA_CHECK(cudaMemcpy(d_buf, h_pageable, bytes, cudaMemcpyHostToDevice));
            processKernel<<<GRID, BLOCK>>>(d_compute, 1.0001f, N, ITERS);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        double ms = (wall_ms() - t0) / REPS;
        printf("  Wall time per iteration : %.2f ms\n", ms);
        printf("  (Copy and compute are strictly sequential)\n");
    }

    // ── Case B: async copy with pageable memory — the silent failure ──────────
    section("Case B: cudaMemcpyAsync with PAGEABLE memory (silent sync!)");
    {
        cudaStream_t copyStream, computeStream;
        CUDA_CHECK(cudaStreamCreateWithFlags(&copyStream,    cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking));

        processKernel<<<GRID, BLOCK, 0, computeStream>>>(d_compute, 1.0001f, N, ITERS);
        CUDA_CHECK(cudaDeviceSynchronize());

        double t0 = wall_ms();
        for (int r = 0; r < REPS; r++) {
            // The API accepts this without error — but the copy engine
            // internally stages through a pinned bounce buffer, making it
            // effectively synchronous from the GPU's point of view.
            CUDA_CHECK(cudaMemcpyAsync(d_buf, h_pageable, bytes,
                                       cudaMemcpyHostToDevice, copyStream));
            processKernel<<<GRID, BLOCK, 0, computeStream>>>(d_compute, 1.0001f, N, ITERS);

            CUDA_CHECK(cudaStreamSynchronize(copyStream));
            CUDA_CHECK(cudaStreamSynchronize(computeStream));
        }
        double ms = (wall_ms() - t0) / REPS;
        printf("  Wall time per iteration : %.2f ms\n", ms);
        printf("  (Looks async in code, but GPU timeline is still sequential)\n");
        printf("  → No speedup over Case A because pageable memory can't DMA directly\n");

        CUDA_CHECK(cudaStreamDestroy(copyStream));
        CUDA_CHECK(cudaStreamDestroy(computeStream));
    }

    // ── Case C: async copy with PINNED memory — genuine overlap ──────────────
    section("Case C: cudaMemcpyAsync with PINNED memory (genuine overlap)");
    {
        cudaStream_t copyStream, computeStream;
        CUDA_CHECK(cudaStreamCreateWithFlags(&copyStream,    cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking));

        processKernel<<<GRID, BLOCK, 0, computeStream>>>(d_compute, 1.0001f, N, ITERS);
        CUDA_CHECK(cudaDeviceSynchronize());

        double t0 = wall_ms();
        for (int r = 0; r < REPS; r++) {
            // The DMA engine can access pinned memory's stable physical address
            // directly, without CPU involvement after the transfer is initiated.
            // The copy engine and the SM compute engine run in parallel.
            CUDA_CHECK(cudaMemcpyAsync(d_buf, h_pinned, bytes,
                                       cudaMemcpyHostToDevice, copyStream));
            processKernel<<<GRID, BLOCK, 0, computeStream>>>(d_compute, 1.0001f, N, ITERS);

            CUDA_CHECK(cudaStreamSynchronize(copyStream));
            CUDA_CHECK(cudaStreamSynchronize(computeStream));
        }
        double ms = (wall_ms() - t0) / REPS;
        printf("  Wall time per iteration : %.2f ms\n", ms);
        printf("  (Copy and compute overlap → total time ≈ max of both)\n");

        CUDA_CHECK(cudaStreamDestroy(copyStream));
        CUDA_CHECK(cudaStreamDestroy(computeStream));
    }

    printf("\n");
    printf("Summary\n");
    printf("-------\n");
    printf("cudaMemcpyAsync requires PINNED host memory to actually be async.\n");
    printf("Allocate host buffers with cudaMallocHost(), free with cudaFreeHost().\n");
    printf("Regular malloc() silently degrades to synchronous transfer.\n\n");
    printf("Verify with Nsight Systems — look for overlapping copy and compute bars:\n");
    printf("  nsys profile --trace=cuda ./build/02_async_copy\n");

    free(h_pageable);
    CUDA_CHECK(cudaFreeHost(h_pinned));
    cudaFree(d_buf); cudaFree(d_compute);
    return 0;
}
