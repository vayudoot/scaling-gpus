// occupancy.cu — Post 2: Your First CUDA Kernels
//
// Demonstrates:
//   • What occupancy is: active warps / maximum warps per SM
//   • How to query the CUDA occupancy calculator
//   • How block size choices affect theoretical and achieved occupancy
//   • Why 256 threads/block is usually a safe default

#include <cuda_runtime.h>
#include <stdio.h>
#include "../include/timer.cuh"

// ─────────────────────────────────────────────────────────────────────────────
// A simple kernel with controllable register pressure
// ─────────────────────────────────────────────────────────────────────────────
// The 'work' parameter is a loop count. Larger work → more register usage
// (the compiler keeps loop variables in registers), which can reduce occupancy.

__global__ void dummyKernel(float* data, int n, int work) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float acc = data[i];
    // Artificial work to vary register usage
    for (int w = 0; w < work; w++) acc = acc * 1.0001f + 0.0001f;
    data[i] = acc;
}

// ─────────────────────────────────────────────────────────────────────────────
// Print occupancy for a range of block sizes
// ─────────────────────────────────────────────────────────────────────────────

void printOccupancyTable(int n, int smem_bytes = 0) {
    printf("Block size | Warps/block | Max blocks/SM | Theor. occupancy\n");
    printf("-----------|-------------|---------------|------------------\n");

    int blockSizes[] = {32, 64, 128, 256, 512, 1024};
    for (int bs : blockSizes) {
        int minGridSize, bestBlockSize;
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &bestBlockSize,
                                           dummyKernel, smem_bytes, 0);

        int maxActiveBlocks;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxActiveBlocks, dummyKernel, bs, smem_bytes);

        // Theoretical occupancy = (active warps) / (max warps per SM)
        // Max warps per SM = 64 on Ampere/Hopper
        int warpsPerBlock   = (bs + 31) / 32;
        int activeWarps     = maxActiveBlocks * warpsPerBlock;
        float occupancy     = (float)activeWarps / 64.f;  // 64 warps/SM on Hopper

        const char* star = (bs == bestBlockSize) ? " ← recommended" : "";
        printf("  %4d     |     %2d      |      %2d        |   %.0f%%%s\n",
               bs, warpsPerBlock, maxActiveBlocks,
               occupancy * 100.f, star);
    }
    printf("\n");
}

// ─────────────────────────────────────────────────────────────────────────────
// Show how occupancy affects latency hiding
// ─────────────────────────────────────────────────────────────────────────────
//
// When a warp stalls waiting for global memory (~600 cycles on H100), the
// warp scheduler switches to another ready warp. High occupancy means more
// warps available to switch to, hiding more latency.
//
// This simple experiment measures kernel time vs block size for a
// memory-bandwidth-bound workload (vector scale) to show the occupancy effect.

__global__ void scaleKernel(float* data, float scalar, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] *= scalar;
}

void benchmarkBlockSizes(float* d_data, int n) {
    int blockSizes[] = {32, 64, 128, 256, 512, 1024};
    printf("Block size | Kernel time (ms) | Eff. BW (GB/s)\n");
    printf("-----------|------------------|----------------\n");

    for (int bs : blockSizes) {
        int grid = (n + bs - 1) / bs;

        // Warm up
        scaleKernel<<<grid, bs>>>(d_data, 1.0001f, n);
        CUDA_CHECK(cudaDeviceSynchronize());

        const int REPS = 20;
        GpuTimer t;
        t.start();
        for (int r = 0; r < REPS; r++)
            scaleKernel<<<grid, bs>>>(d_data, 1.0001f, n);
        float ms = t.stop_ms() / REPS;

        // BW: read + write = 2 × n × 4 bytes
        double bw = 2.0 * n * sizeof(float) / (ms * 1e-3) / 1e9;
        printf("  %4d     |     %.3f ms       |   %.1f\n", bs, ms, bw);
    }
    printf("\n");
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    print_device_info();

    int n = 1 << 24;   // 16 M elements

    printf("═══ Theoretical occupancy vs block size ═══\n");
    printf("(for dummyKernel with 0 explicit shared memory)\n\n");
    printOccupancyTable(n, 0);

    printf("═══ Observed bandwidth vs block size ═══\n");
    printf("(scaleKernel on %d floats — memory-bound workload)\n\n", n);

    float* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_data, 0, n * sizeof(float)));

    benchmarkBlockSizes(d_data, n);

    printf("Insight: for memory-bound kernels, block sizes from 128–512 tend\n"
           "to produce similar bandwidth because the GPU can hide HBM latency\n"
           "by switching between resident warps. Very small blocks (32, 64)\n"
           "reduce occupancy and leave latency exposed, lowering throughput.\n");

    cudaFree(d_data);
    return 0;
}
