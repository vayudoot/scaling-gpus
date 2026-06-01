// 01_coalescing.cu — Post 3: GPU Memory: The Real Bottleneck
//
// Demonstrates memory coalescing — the single most impactful memory access
// pattern rule in CUDA.
//
// When all 32 threads in a warp read consecutive addresses, the hardware
// issues ONE 128-byte memory transaction.  When they read strided addresses,
// it issues up to 32 separate transactions — 32× more bus traffic for the
// same useful data.
//
// This program benchmarks four access patterns:
//   1. Coalesced       — threads read array[threadId]          (stride-1)
//   2. Strided-2       — threads read array[threadId * 2]      (stride-2)
//   3. Strided-8       — threads read array[threadId * 8]      (stride-8)
//   4. Strided-32      — threads read array[threadId * 32]     (one 128B line per thread)
//   5. Random          — threads read via a shuffled index array
//
// Expected result: strided and random kernels achieve a small fraction of
// the coalesced kernel's bandwidth, even though they do identical arithmetic.

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/timer.cuh"

// ─────────────────────────────────────────────────────────────────────────────
// Kernels
// ─────────────────────────────────────────────────────────────────────────────

// Stride-1: all 32 threads in a warp access consecutive floats.
// Hardware issues 1 × 128-byte transaction per warp.  Full bandwidth.
__global__ void coalesced(const float* __restrict__ in,
                           float*       __restrict__ out,
                           int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * 2.0f;
}

// Stride-S: thread i accesses in[i * stride].
// For stride=32: threads 0,1,2,... access bytes 0,128,256,...
// Each access falls in a different 128-byte cache line → 32 transactions.
__global__ void strided(const float* __restrict__ in,
                         float*       __restrict__ out,
                         int n, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = i * stride;
    if (idx < n) out[idx] = in[idx] * 2.0f;
}

// Random: thread i accesses in[indices[i]] — worst case scatter.
// Every warp issues up to 32 separate transactions.
__global__ void random_access(const float* __restrict__ in,
                               float*       __restrict__ out,
                               const int*   __restrict__ indices,
                               int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int idx = indices[i];
        out[idx] = in[idx] * 2.0f;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Fisher-Yates shuffle to build a random permutation
// ─────────────────────────────────────────────────────────────────────────────
static void shuffle(int* arr, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark helper
// ─────────────────────────────────────────────────────────────────────────────
template<typename Fn>
static float bench_ms(Fn fn, int reps = 20) {
    fn(); CUDA_CHECK(cudaDeviceSynchronize()); // warm up
    GpuTimer t; t.start();
    for (int r = 0; r < reps; r++) fn();
    return t.stop_ms() / reps;
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    print_device_info();

    // n must be large enough to give the GPU many warps to schedule.
    // stride-32 touches n*32 addresses so we need n*32 < total allocation.
    int n = (argc > 1) ? atoi(argv[1]) : (1 << 22);   // 4 M elements default
    // stride-32 kernel: thread i accesses index i*32, so max index = (n-1)*32.
    // Allocate n*32 floats so the strided kernel never goes OOB.
    int n_alloc = n * 32;

    size_t bytes_n      = (size_t)n      * sizeof(float);
    size_t bytes_alloc  = (size_t)n_alloc * sizeof(float);

    printf("Elements (coalesced/random) : %d  (%.1f MB)\n", n,
           (double)bytes_n / 1e6);
    printf("Physical allocation         : %d  (%.1f MB)\n\n", n_alloc,
           (double)bytes_alloc / 1e6);

    // ── Allocate ─────────────────────────────────────────────────────────────
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  bytes_alloc));
    CUDA_CHECK(cudaMalloc(&d_out, bytes_alloc));
    CUDA_CHECK(cudaMemset(d_in,  0, bytes_alloc));
    CUDA_CHECK(cudaMemset(d_out, 0, bytes_alloc));

    // Build shuffled index array for random-access kernel
    int* h_idx = (int*)malloc(bytes_n);
    for (int i = 0; i < n; i++) h_idx[i] = i;
    srand(42); shuffle(h_idx, n);
    int* d_idx;
    CUDA_CHECK(cudaMalloc(&d_idx, bytes_n));
    CUDA_CHECK(cudaMemcpy(d_idx, h_idx, bytes_n, cudaMemcpyHostToDevice));

    int block = 256;
    int grid  = (n + block - 1) / block;

    // ── Table header ─────────────────────────────────────────────────────────
    printf("%-22s %10s %12s %16s\n",
           "Kernel", "Time (ms)", "BW (GB/s)", "vs coalesced");
    printf("%-22s %10s %12s %16s\n",
           "──────────────────────", "─────────", "──────────", "────────────");

    float ms;
    double bw_coal;

    // 1. Coalesced — stride 1
    ms = bench_ms([&](){ coalesced<<<grid, block>>>(d_in, d_out, n); });
    bw_coal = bandwidth_gb_s(2ull * n * sizeof(float), ms);
    printf("%-22s %10.3f %12.1f  ← baseline\n", "coalesced (stride-1)", ms, bw_coal);

    // 2. Strided-2
    ms = bench_ms([&](){ strided<<<grid, block>>>(d_in, d_out, n_alloc, 2); });
    { double bw = bandwidth_gb_s(2ull * n * sizeof(float), ms);
      printf("%-22s %10.3f %12.1f  (%.0f%% of coalesced)\n",
             "strided (stride-2)", ms, bw, bw/bw_coal*100); }

    // 3. Strided-8
    ms = bench_ms([&](){ strided<<<grid, block>>>(d_in, d_out, n_alloc, 8); });
    { double bw = bandwidth_gb_s(2ull * n * sizeof(float), ms);
      printf("%-22s %10.3f %12.1f  (%.0f%% of coalesced)\n",
             "strided (stride-8)", ms, bw, bw/bw_coal*100); }

    // 4. Strided-32  (worst-case strided — every thread in its own cache line)
    ms = bench_ms([&](){ strided<<<grid, block>>>(d_in, d_out, n_alloc, 32); });
    { double bw = bandwidth_gb_s(2ull * n * sizeof(float), ms);
      printf("%-22s %10.3f %12.1f  (%.0f%% of coalesced)\n",
             "strided (stride-32)", ms, bw, bw/bw_coal*100); }

    // 5. Random
    ms = bench_ms([&](){ random_access<<<grid, block>>>(d_in, d_out, d_idx, n); });
    { double bw = bandwidth_gb_s(2ull * n * sizeof(float), ms);
      printf("%-22s %10.3f %12.1f  (%.0f%% of coalesced)\n",
             "random", ms, bw, bw/bw_coal*100); }

    printf("\n");
    printf("Key insight\n");
    printf("-----------\n");
    printf("All kernels perform identical arithmetic (one multiply per element).\n");
    printf("Throughput difference comes entirely from memory transaction count:\n");
    printf("  coalesced: 1 transaction per 32 threads (128 B, 32 floats used)\n");
    printf("  stride-32: 32 transactions per 32 threads (4096 B fetched, 128 B used)\n");
    printf("  random:    up to 32 transactions per 32 threads\n\n");
    printf("Rule: threads in a warp should always access consecutive addresses.\n");

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_idx);
    free(h_idx);
    return 0;
}
