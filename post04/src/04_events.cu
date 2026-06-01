// 04_events.cu — Post 4: CUDA Streams and Async Execution
//
// CUDA events are the precision tool for two jobs:
//
//   1. TIMING: measuring GPU execution time without stalling the CPU.
//      cudaEventRecord() inserts a timestamp into a stream's command queue.
//      The GPU writes the timestamp when it reaches that point in the queue.
//      cudaEventElapsedTime() computes the delta — accurate to ~0.5µs.
//
//   2. CROSS-STREAM SYNCHRONIZATION: making one stream wait for a specific
//      point in another stream, WITHOUT the CPU being involved.
//      cudaStreamWaitEvent() inserts a dependency fence into the target stream
//      that blocks only that stream until the event fires — the CPU continues.
//
// Why events beat cudaStreamSynchronize for cross-stream deps:
//   cudaStreamSynchronize(s) → blocks the CPU thread until stream s is done.
//                              The CPU cannot launch more work while waiting.
//   cudaStreamWaitEvent(s, e) → the CPU returns immediately.
//                               Stream s pauses on the GPU until event e fires.
//                               The CPU can submit more work to other streams.
//
// This program demonstrates all three uses:
//   A. Event-based kernel timing
//   B. GPU-side cross-stream dependency via events (CPU-free)
//   C. The common mistake: using cudaDeviceSynchronize in a hot path

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "../include/timer.cuh"

// ─────────────────────────────────────────────────────────────────────────────
// Kernels
// ─────────────────────────────────────────────────────────────────────────────

__global__ void stage1Kernel(float* data, int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = (float)i;
    for (int it = 0; it < iters; it++) v = v * 1.0001f + 0.0001f;
    data[i] = v;
}

// Stage 2 READS from stage1's output — must not start before stage1 finishes.
__global__ void stage2Kernel(const float* __restrict__ in,
                              float*       __restrict__ out,
                              int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = in[i];
    for (int it = 0; it < iters; it++) v = sqrtf(v) + 0.001f;
    out[i] = v;
}

// Independent kernel that can run concurrently with stage1
__global__ void independentKernel(float* data, int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = (float)(n - i);
    for (int it = 0; it < iters; it++) v = v * 1.0002f - 0.0001f;
    data[i] = v;
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    print_device_info();

    const int N     = 1 << 22;   // 4 M floats
    const int ITERS = 150;
    const int BLOCK = 256;
    const int GRID  = (N + BLOCK - 1) / BLOCK;

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(float)));

    // ── A: Event-based kernel timing ──────────────────────────────────────────
    section("A: Event-based GPU timing");
    {
        // CPU timers measure wall time including scheduling jitter.
        // GPU events are inserted into the command queue and timestamped by
        // the GPU itself — they measure actual hardware execution time.
        cudaEvent_t e_start, e_stop;
        CUDA_CHECK(cudaEventCreate(&e_start));
        CUDA_CHECK(cudaEventCreate(&e_stop));

        // Warm up
        stage1Kernel<<<GRID, BLOCK>>>(d_a, N, ITERS);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Record start event → inserted into stream 0's queue
        CUDA_CHECK(cudaEventRecord(e_start));

        stage1Kernel<<<GRID, BLOCK>>>(d_a, N, ITERS);

        // Record stop event → GPU writes timestamp when it reaches this point
        CUDA_CHECK(cudaEventRecord(e_stop));

        // CPU blocks until e_stop is written (i.e., kernel is done)
        CUDA_CHECK(cudaEventSynchronize(e_stop));

        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e_start, e_stop));
        printf("  stage1Kernel execution time : %.3f ms\n", ms);
        printf("  (GPU-measured, ±0.5µs accuracy)\n\n");

        // You can also time a sequence of operations:
        CUDA_CHECK(cudaEventRecord(e_start));
        stage1Kernel<<<GRID, BLOCK>>>(d_a, N, ITERS);
        stage2Kernel<<<GRID, BLOCK>>>(d_a, d_b, N, ITERS / 2);
        CUDA_CHECK(cudaEventRecord(e_stop));
        CUDA_CHECK(cudaEventSynchronize(e_stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms, e_start, e_stop));
        printf("  stage1 + stage2 combined    : %.3f ms\n", ms);

        CUDA_CHECK(cudaEventDestroy(e_start));
        CUDA_CHECK(cudaEventDestroy(e_stop));
    }

    // ── B: Cross-stream GPU-side dependency via events ────────────────────────
    //
    // Scenario: we have three pieces of work:
    //   stage1(d_a → d_b)    — stream A
    //   independent(d_c)     — stream B (no dependency on stage1)
    //   stage2(d_b → d_a)    — stream A or B, but MUST wait for stage1
    //
    // We want: stage1 and independentKernel to run concurrently.
    //          stage2 to run only after stage1 finishes (not after independent).
    //
    // With cudaStreamWaitEvent, we express exactly this dependency.
    // The CPU submits ALL three launches and returns immediately.
    // The GPU enforces the ordering itself.

    section("B: GPU-side cross-stream synchronisation with events");
    {
        cudaStream_t sA, sB;
        CUDA_CHECK(cudaStreamCreateWithFlags(&sA, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&sB, cudaStreamNonBlocking));

        cudaEvent_t stage1_done;
        CUDA_CHECK(cudaEventCreate(&stage1_done));

        CUDA_CHECK(cudaDeviceSynchronize());  // clean start

        // ── CPU submits all work in one go ────────────────────────────────────
        double t_cpu_start = wall_ms();

        // Stream A: run stage1, then record that it's done
        stage1Kernel<<<GRID, BLOCK, 0, sA>>>(d_a, N, ITERS);
        CUDA_CHECK(cudaEventRecord(stage1_done, sA));
        // stage1_done fires on the GPU when sA reaches this point

        // Stream B: run an independent kernel (no dependency on stage1)
        independentKernel<<<GRID, BLOCK, 0, sB>>>(d_c, N, ITERS);

        // Stream B must wait for stage1_done before it runs stage2.
        // This is a GPU-side fence — the CPU returns immediately.
        // Stream B pauses internally until the GPU signals stage1_done.
        CUDA_CHECK(cudaStreamWaitEvent(sB, stage1_done, 0));
        stage2Kernel<<<GRID, BLOCK, 0, sB>>>(d_a, d_b, N, ITERS / 2);

        double t_cpu_submit = wall_ms() - t_cpu_start;
        printf("  CPU time to submit all work : %.3f ms\n", t_cpu_submit);
        printf("  (CPU returned while GPU was still running — true async)\n\n");

        // Now wait for all GPU work to complete
        cudaEvent_t all_done;
        CUDA_CHECK(cudaEventCreate(&all_done));
        CUDA_CHECK(cudaEventRecord(all_done, sB));
        CUDA_CHECK(cudaEventSynchronize(all_done));

        // Time the whole pipeline
        float ms_gpu;
        cudaEvent_t t_start, t_stop;
        CUDA_CHECK(cudaEventCreate(&t_start));
        CUDA_CHECK(cudaEventCreate(&t_stop));

        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(t_start, sA));

        stage1Kernel<<<GRID, BLOCK, 0, sA>>>(d_a, N, ITERS);
        CUDA_CHECK(cudaEventRecord(stage1_done, sA));

        independentKernel<<<GRID, BLOCK, 0, sB>>>(d_c, N, ITERS);
        CUDA_CHECK(cudaStreamWaitEvent(sB, stage1_done, 0));
        stage2Kernel<<<GRID, BLOCK, 0, sB>>>(d_a, d_b, N, ITERS / 2);

        CUDA_CHECK(cudaEventRecord(t_stop, sB));
        CUDA_CHECK(cudaEventSynchronize(t_stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms_gpu, t_start, t_stop));

        printf("  Full pipeline GPU time      : %.3f ms\n", ms_gpu);
        printf("  (stage1 ∥ independent, then stage2 after stage1 only)\n\n");

        printf("  Dependency graph:\n");
        printf("    sA: [stage1] ──(stage1_done)──────────────────────────\n");
        printf("    sB: [independent]───────────── wait ──── [stage2]\n");
        printf("         ↑ runs concurrently with stage1\n\n");

        printf("  Key: cudaStreamWaitEvent(sB, stage1_done) puts the fence\n");
        printf("  inside sB's queue — the CPU did not block at all.\n");

        CUDA_CHECK(cudaEventDestroy(stage1_done));
        CUDA_CHECK(cudaEventDestroy(all_done));
        CUDA_CHECK(cudaEventDestroy(t_start));
        CUDA_CHECK(cudaEventDestroy(t_stop));
        CUDA_CHECK(cudaStreamDestroy(sA));
        CUDA_CHECK(cudaStreamDestroy(sB));
    }

    // ── C: The cudaDeviceSynchronize anti-pattern ─────────────────────────────
    section("C: Why cudaDeviceSynchronize() is wrong in a hot path");
    {
        cudaStream_t sA, sB;
        CUDA_CHECK(cudaStreamCreateWithFlags(&sA, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&sB, cudaStreamNonBlocking));

        // Wrong: using cudaDeviceSynchronize between stages
        // This stalls the CPU AND collapses all stream concurrency.
        CUDA_CHECK(cudaDeviceSynchronize());
        double t0 = wall_ms();
        {
            stage1Kernel<<<GRID, BLOCK, 0, sA>>>(d_a, N, ITERS);
            CUDA_CHECK(cudaDeviceSynchronize()); // ← CPU blocks, sB also stalls
            stage2Kernel<<<GRID, BLOCK, 0, sB>>>(d_a, d_b, N, ITERS / 2);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        double ms_bad = wall_ms() - t0;

        // Right: event-based dependency
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEvent_t done;
        CUDA_CHECK(cudaEventCreate(&done));

        t0 = wall_ms();
        {
            stage1Kernel<<<GRID, BLOCK, 0, sA>>>(d_a, N, ITERS);
            CUDA_CHECK(cudaEventRecord(done, sA));
            CUDA_CHECK(cudaStreamWaitEvent(sB, done, 0)); // GPU-side only
            stage2Kernel<<<GRID, BLOCK, 0, sB>>>(d_a, d_b, N, ITERS / 2);
            CUDA_CHECK(cudaEventSynchronize(done));
            CUDA_CHECK(cudaStreamSynchronize(sB));
        }
        double ms_good = wall_ms() - t0;

        printf("  cudaDeviceSynchronize() between stages : %.2f ms\n", ms_bad);
        printf("  cudaEventRecord + cudaStreamWaitEvent  : %.2f ms\n", ms_good);
        printf("\n  cudaDeviceSynchronize() stalls the CPU and serialises ALL streams.\n");
        printf("  Events stall only the specific stream that needs the dependency.\n");
        printf("  Use cudaDeviceSynchronize() only for error checking and teardown.\n");

        CUDA_CHECK(cudaEventDestroy(done));
        CUDA_CHECK(cudaStreamDestroy(sA));
        CUDA_CHECK(cudaStreamDestroy(sB));
    }

    printf("\nTo see event markers in Nsight Systems:\n");
    printf("  nsys profile --trace=cuda,nvtx ./build/04_events\n");

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}
