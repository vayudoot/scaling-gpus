// 03_ncu_target.cu  --  Post 11: Profiling and Optimization
//
// A Nsight Compute practice target: the SAME reduction (sum of 64M floats)
// implemented three ways, each with a known, distinct bottleneck. Run ncu on
// each kernel and match what you see to the expected metric signature --
// this is the fastest way to learn to read an ncu report.
//
//   A. atomicAllKernel     every element does a global atomicAdd
//                          -> bottleneck: atomic serialization
//   B. stridedKernel       block-local reduction, but each thread reads a
//                          CONTIGUOUS PRIVATE CHUNK -> warp accesses are
//                          strided across memory -> uncoalesced loads
//   C. coalescedKernel     grid-stride loop + shared-memory tree + one
//                          atomic per block -> the healthy version
//
// All three verified against a CPU sum. Bandwidth printed per variant.
//
// Then run (from the post's Nsight Compute section):
//   ncu --kernel-name atomicAllKernel --set full ./build/03_ncu_target
//   ncu --kernel-name stridedKernel   --set full ./build/03_ncu_target
//   ncu --kernel-name coalescedKernel --set full ./build/03_ncu_target
//
// What to look for (metric names from the post):
//   variant  dram__throughput   l1tex...sectors/req    smsp__..._per_warp_stalls
//   A        LOW                ~1 (coalesced)         huge LG throttle / atomics
//   B        LOW-MED            ~32 (uncoalesced!)     long scoreboard (memory)
//   C        HIGH (near peak)   ~4 (coalesced)         few stalls -- BW-bound
//
// The lesson: all three are "memory" kernels, but ncu tells you WHICH memory
// problem you have -- serialization, access pattern, or none at all.

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/utils.cuh"

// A: everyone hammers one address
__global__ void atomicAllKernel(const float* __restrict__ x,
                                float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(out, x[i]);
}

// B: each thread reduces its own contiguous chunk -> strided across the warp
__global__ void stridedKernel(const float* __restrict__ x,
                              float* __restrict__ out, int n, int chunk) {
    __shared__ float s[256];
    int tid  = threadIdx.x;
    int t    = blockIdx.x * blockDim.x + threadIdx.x;
    long long beg = (long long)t * chunk;
    float acc = 0.f;
    for (int k = 0; k < chunk; k++) {
        long long idx = beg + k;
        if (idx < n) acc += x[idx];      // thread-contiguous = warp-strided!
    }
    s[tid] = acc; __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, s[0]);
}

// C: grid-stride (warp-contiguous) + shared tree + one atomic per block
__global__ void coalescedKernel(const float* __restrict__ x,
                                float* __restrict__ out, int n) {
    __shared__ float s[256];
    int tid = threadIdx.x;
    float acc = 0.f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x)
        acc += x[i];                     // consecutive threads -> consecutive addrs
    s[tid] = acc; __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, s[0]);
}

int main() {
    print_device_info();
    srand(42);

    const int N   = 64 * 1024 * 1024;      // 256 MB of floats
    const int BLK = 256;
    size_t bytes  = (size_t)N * sizeof(float);

    float* h_x = (float*)malloc(bytes);
    // small values so the FP32 atomic orderings stay comparable to CPU double
    rand_fill(h_x, N, 0.f, 1e-3f);
    double cpu_sum = 0.0;
    for (int i = 0; i < N; i++) cpu_sum += h_x[i];

    float *d_x, *d_out;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));

    printf("Reducing %d floats (%.0f MB); CPU reference sum = %.4f\n\n",
           N, bytes / 1e6, cpu_sum);
    printf("  %-22s %12s %12s %12s  %s\n",
           "kernel", "ms", "GB/s", "sum", "check");

    // helper pattern: warm, time, verify -- written out per variant
    // (no timing macros around <<<>>> -- see the series build rules)

    // ── A ─────────────────────────────────────────────────────────────────────
    {
        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
        atomicAllKernel<<<(N+BLK-1)/BLK, BLK>>>(d_x, d_out, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
        GpuTimer t; t.start();
        atomicAllKernel<<<(N+BLK-1)/BLK, BLK>>>(d_x, d_out, N);
        float ms = t.stop_ms();
        float s; CUDA_CHECK(cudaMemcpy(&s, d_out, 4, cudaMemcpyDeviceToHost));
        // NOTE: 64M float adds into ONE float accumulator -- late increments
        // fall below the accumulator's ULP, so a few % drift is EXPECTED.
        // That drift is a Post 7 lesson showing up uninvited.
        printf("  %-22s %12.3f %12.1f %12.4f  %s\n", "A atomicAll",
               ms, bw_gb_s(bytes, ms), s,
               fabs(s - cpu_sum) < 0.10 * cpu_sum ? "ok (float drift expected)"
                                                  : "OFF");
    }

    // ── B ─────────────────────────────────────────────────────────────────────
    {
        const int CHUNK = 64;
        int threads_total = (N + CHUNK - 1) / CHUNK;
        int grid = (threads_total + BLK - 1) / BLK;
        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
        stridedKernel<<<grid, BLK>>>(d_x, d_out, N, CHUNK);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
        GpuTimer t; t.start();
        stridedKernel<<<grid, BLK>>>(d_x, d_out, N, CHUNK);
        float ms = t.stop_ms();
        float s; CUDA_CHECK(cudaMemcpy(&s, d_out, 4, cudaMemcpyDeviceToHost));
        printf("  %-22s %12.3f %12.1f %12.4f  %s\n", "B strided(chunk=64)",
               ms, bw_gb_s(bytes, ms), s,
               fabs(s - cpu_sum) < 1e-2 * cpu_sum ? "ok" : "OFF");
    }

    // ── C ─────────────────────────────────────────────────────────────────────
    {
        int grid = 4 * 128;   // enough blocks to saturate; grid-stride covers N
        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
        coalescedKernel<<<grid, BLK>>>(d_x, d_out, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
        GpuTimer t; t.start();
        coalescedKernel<<<grid, BLK>>>(d_x, d_out, N);
        float ms = t.stop_ms();
        float s; CUDA_CHECK(cudaMemcpy(&s, d_out, 4, cudaMemcpyDeviceToHost));
        printf("  %-22s %12.3f %12.1f %12.4f  %s\n", "C coalesced",
               ms, bw_gb_s(bytes, ms), s,
               fabs(s - cpu_sum) < 1e-2 * cpu_sum ? "ok" : "OFF");
    }

    section("Now point Nsight Compute at each one");
    printf("  ncu --kernel-name atomicAllKernel --set full ./build/03_ncu_target\n");
    printf("  ncu --kernel-name stridedKernel   --set full ./build/03_ncu_target\n");
    printf("  ncu --kernel-name coalescedKernel --set full ./build/03_ncu_target\n\n");
    printf("  Expected signatures (the post's metric table, in the wild):\n");
    printf("  %-4s %-24s %-26s %-22s\n", "", "dram__throughput", "l1tex sectors/request", "dominant stall");
    printf("  %-4s %-24s %-26s %-22s\n", "A", "LOW (single hot addr)", "low (reads coalesced)", "LG throttle / atomics");
    printf("  %-4s %-24s %-26s %-22s\n", "B", "LOW-MED", "high (~32: uncoalesced)", "long scoreboard");
    printf("  %-4s %-24s %-26s %-22s\n", "C", "HIGH (near peak)", "~4 (coalesced)", "(few) -- BW-bound");
    printf("\n  Same math, three different diagnoses. ncu is how you tell them apart\n");
    printf("  without guessing -- and Post 3's coalescing rules are the fix.\n");

    cudaFree(d_x); cudaFree(d_out); free(h_x);
    return 0;
}
