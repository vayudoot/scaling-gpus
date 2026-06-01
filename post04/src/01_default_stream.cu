// 01_default_stream.cu — Post 4: CUDA Streams and Async Execution
//
// The default stream (stream 0) has a special property: it is SYNCHRONIZING.
// No other operation can run on the GPU while the default stream is active,
// and the default stream waits for all other streams before it starts.
//
// This program measures the difference between:
//   A. Running two independent kernels on the DEFAULT stream → sequential
//   B. Running the same two kernels on DIFFERENT streams    → concurrent
//
// The kernels are independent (no data dependency), so concurrent execution
// should take roughly half the wall time of sequential execution — assuming
// the GPU has enough SMs to schedule both simultaneously.
//
// What to observe:
//   - Default stream: total time ≈ sum of individual times
//   - Two streams:    total time ≈ max of individual times (if they overlap)
//
// Verify in Nsight Systems:
//   nsys profile --trace=cuda ./build/01_default_stream
//   Look for the two kernel bars — with separate streams they should overlap.

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "../include/timer.cuh"

// ─────────────────────────────────────────────────────────────────────────────
// A simple compute kernel: elementwise multiply-add repeated 'iters' times.
// The 'iters' parameter lets us tune how long the kernel runs (its "width"
// on the Nsight Systems timeline) independently of grid/block size.
// ─────────────────────────────────────────────────────────────────────────────

__global__ void computeKernel(float* data, float scalar, int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = data[i];
    // Repeated work to keep the kernel running long enough to see in a timeline
    for (int it = 0; it < iters; it++)
        v = v * scalar + (1.0f - scalar);
    data[i] = v;
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    print_device_info();

    // Two independent datasets — kernelA operates on d0, kernelB on d1
    const int N     = 1 << 22;   // 4 M floats each
    const int ITERS = 200;        // enough work to make overlap visible
    const int BLOCK = 256;
    const int GRID  = (N + BLOCK - 1) / BLOCK;

    float *d0, *d1;
    CUDA_CHECK(cudaMalloc(&d0, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d1, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d0, 1, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d1, 1, N * sizeof(float)));

    // ── Measure individual kernel times ──────────────────────────────────────
    section("Individual kernel timing (for reference)");
    {
        // Warm up
        computeKernel<<<GRID, BLOCK>>>(d0, 1.0001f, N, ITERS);
        CUDA_CHECK(cudaDeviceSynchronize());

        GpuTimer t;
        t.start();
        computeKernel<<<GRID, BLOCK>>>(d0, 1.0001f, N, ITERS);
        float msA = t.stop_ms();

        t.start();
        computeKernel<<<GRID, BLOCK>>>(d1, 1.0002f, N, ITERS);
        float msB = t.stop_ms();

        printf("  Kernel A alone : %.2f ms\n", msA);
        printf("  Kernel B alone : %.2f ms\n", msB);
        printf("  Sum (expected if sequential) : %.2f ms\n", msA + msB);
    }

    // ── Default stream: both kernels on stream 0 ──────────────────────────────
    // The default stream is synchronising: kernelB cannot start until kernelA
    // has finished, even though they are completely independent.
    section("Default stream (synchronising — kernels run sequentially)");
    {
        CUDA_CHECK(cudaDeviceSynchronize());
        GpuTimer t; t.start();

        computeKernel<<<GRID, BLOCK>>>(d0, 1.0001f, N, ITERS);  // stream 0
        computeKernel<<<GRID, BLOCK>>>(d1, 1.0002f, N, ITERS);  // stream 0

        float ms = t.stop_ms();
        printf("  Both on default stream : %.2f ms\n", ms);
        printf("  (Expected: ≈ sum of individual times)\n");
    }

    // ── Two non-default streams: kernels may run concurrently ─────────────────
    // With cudaStreamNonBlocking, these streams are fully independent of stream 0
    // and of each other.  The GPU scheduler can run both kernels simultaneously
    // as long as there are enough SMs to hold both warps.
    section("Two non-blocking streams (kernels can overlap)");
    {
        cudaStream_t sA, sB;
        CUDA_CHECK(cudaStreamCreateWithFlags(&sA, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&sB, cudaStreamNonBlocking));

        CUDA_CHECK(cudaDeviceSynchronize());
        GpuTimer t; t.start();

        computeKernel<<<GRID, BLOCK, 0, sA>>>(d0, 1.0001f, N, ITERS);
        computeKernel<<<GRID, BLOCK, 0, sB>>>(d1, 1.0002f, N, ITERS);

        // Wait for BOTH streams to complete
        CUDA_CHECK(cudaStreamSynchronize(sA));
        CUDA_CHECK(cudaStreamSynchronize(sB));
        float ms = t.stop_ms();

        printf("  Both on separate streams : %.2f ms\n", ms);
        printf("  (Expected: ≈ max of individual times if fully overlapping)\n\n");

        printf("  Overlap efficiency : %.0f%%\n",
               100.0f * (1.0f - ms / (/*sum*/ ms)));  // placeholder

        CUDA_CHECK(cudaStreamDestroy(sA));
        CUDA_CHECK(cudaStreamDestroy(sB));
    }

    // ── The NULL stream trap: mixing default and non-default ──────────────────
    // If you put kernelA on a non-default stream but kernelB on the default
    // stream, kernelB will WAIT for kernelA even though you used separate streams.
    // The default stream is the culprit.
    section("The NULL-stream trap: one default + one non-default");
    {
        cudaStream_t sA;
        CUDA_CHECK(cudaStreamCreateWithFlags(&sA, cudaStreamNonBlocking));

        CUDA_CHECK(cudaDeviceSynchronize());
        GpuTimer t; t.start();

        computeKernel<<<GRID, BLOCK, 0, sA>>>(d0, 1.0001f, N, ITERS);
        // Stream 0 (NULL): waits for ALL other streams that were created
        // WITHOUT cudaStreamNonBlocking before it starts.
        // With cudaStreamNonBlocking used for sA, the behaviour depends on
        // driver version and may still serialise on some platforms.
        computeKernel<<<GRID, BLOCK>>>(d1, 1.0002f, N, ITERS);  // NULL stream

        CUDA_CHECK(cudaDeviceSynchronize());
        float ms = t.stop_ms();
        printf("  Non-blocking + NULL stream : %.2f ms\n", ms);
        printf("  Rule: avoid the NULL (default) stream in production kernels.\n");
        printf("  Always pass an explicit stream to every launch and memcpy.\n");

        CUDA_CHECK(cudaStreamDestroy(sA));
    }

    printf("\n");
    printf("To verify overlap visually:\n");
    printf("  nsys profile --trace=cuda ./build/01_default_stream\n");
    printf("  Open the .nsys-rep in Nsight Systems GUI and look at the\n");
    printf("  'CUDA HW' rows — the two kernel bars should overlap for\n");
    printf("  the two-stream case but appear back-to-back for default stream.\n");

    cudaFree(d0); cudaFree(d1);
    return 0;
}
