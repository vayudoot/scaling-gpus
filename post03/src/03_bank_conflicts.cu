// 03_bank_conflicts.cu — Post 3: GPU Memory: The Real Bottleneck
//
// A focused, isolated benchmark for shared memory bank conflicts.
// The transpose program shows bank conflicts in context; this program
// makes them the explicit subject so you can measure the effect cleanly
// and verify with Nsight Compute.
//
// Shared memory has 32 banks, each 4 bytes wide.
// Element at byte offset k lives in bank (k / 4) % 32.
// In a TILE×TILE float array:
//   element [r][c] is at byte offset (r * TILE + c) * 4
//   → bank (r * TILE + c) % 32
//
// When reading tile[threadIdx.x][0] for all 32 threads in a warp:
//   thread 0: tile[0][0]  → bank 0
//   thread 1: tile[1][0]  → bank (TILE) % 32
//                           For TILE=32: bank 0 → 32-way conflict!
//                           For TILE=33: bank 1 → no conflict!
//
// This program implements the same shared-memory reduction with:
//   A. stride-1  access (no conflict)
//   B. stride-16 access (2-way conflict)
//   C. stride-32 access (32-way conflict)
//   D. stride-32 with padding (+1 column) — resolves the conflict
//
// After running, use:
//   ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \
//       ./build/03_bank_conflicts
// to see the raw conflict counts per kernel.

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "../include/timer.cuh"

#define SMEM_SIZE  (32 * 32)   // 32×32 float array = 4 KB
#define BLOCK_SIZE 256

// ─────────────────────────────────────────────────────────────────────────────
// Helper: load data into shared memory then read it back with a given stride.
// A pure shared-memory throughput test with no HBM dependency in the read path.
// ─────────────────────────────────────────────────────────────────────────────

// No conflict: each thread reads its own column — stride 1 within a row.
__global__ void smemStride1(float* out, int n) {
    __shared__ float smem[32][32];

    int tid = threadIdx.x;                       // 0..255
    int lane = tid % 32;                         // position within warp
    int row  = tid / 32;                         // which row

    // Write: each thread writes one element
    smem[row][lane] = (float)(row * 32 + lane);
    __syncthreads();

    // Read stride-1: thread 0 reads [0][0], thread 1 reads [0][1], ...
    // All in different banks → no conflict.
    float val = 0.f;
    for (int iter = 0; iter < 32; iter++)
        val += smem[row][(lane + iter) % 32];  // stride-1 within row

    // Prevent compiler from optimising out the reads
    if (tid < n) out[tid] = val;
}

// 16-way conflict: stride-2 across banks.
// threadIdx.x 0..15 hit banks 0,2,4,...30 — no conflict (all different).
// But threadIdx.x 16..31 hit banks 0,2,4,...30 again → 2-way conflict.
// Using TILE=32: stride-16 means thread i accesses column (i*16)%32.
__global__ void smemStride16(float* out, int n) {
    __shared__ float smem[32][32];

    int tid = threadIdx.x;
    int lane = tid % 32;

    // Write: linear, no conflict
    smem[lane / 32][lane % 32] = (float)tid;
    smem[tid / 32][tid % 32] = (float)tid;
    __syncthreads();

    // Read: stride-16 → some threads share banks
    float val = 0.f;
    for (int iter = 0; iter < 32; iter++)
        val += smem[tid / 32][(lane * 16 + iter) % 32];

    if (tid < n) out[tid] = val;
}

// 32-way conflict: all threads read from the same bank.
// smem[0..31][0] — column 0 of every row.
// With TILE=32: all these are 32 apart → same bank (bank 0) → 32-way conflict.
__global__ void smemStride32(float* out, int n) {
    __shared__ float smem[32][32];

    int tid  = threadIdx.x;
    int lane = tid % 32;

    // Write: no conflict (stride-1)
    smem[lane][tid / 32] = (float)tid;
    __syncthreads();

    // Read: every thread reads column 0 of its row.
    // tile[0][0], tile[1][0], ..., tile[31][0] → all bank 0 → 32-way conflict.
    float val = 0.f;
    for (int iter = 0; iter < 32; iter++)
        val += smem[(lane + iter) % 32][0];   // always column 0 → always bank 0

    if (tid < n) out[tid] = val;
}

// Same access pattern as smemStride32 but with +1 column padding.
// smem[0..31][0]: with TILE=33, row 0 at offset 0 (bank 0),
//                               row 1 at offset 33*4=132 bytes (bank 1),
//                               row 2 at offset 66*4=264 bytes (bank 2), ...
// Each row now starts at a different bank → no conflict.
__global__ void smemStride32Padded(float* out, int n) {
    __shared__ float smem[32][32 + 1];   // ← +1 column padding

    int tid  = threadIdx.x;
    int lane = tid % 32;

    smem[lane][tid / 32] = (float)tid;
    __syncthreads();

    // Same access pattern: column 0 of each row.
    // With padding, each row starts at a different bank → no conflict.
    float val = 0.f;
    for (int iter = 0; iter < 32; iter++)
        val += smem[(lane + iter) % 32][0];

    if (tid < n) out[tid] = val;
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    print_device_info();

    int n = 1 << 20;  // 1 M elements output (just to give many blocks)
    float* d_out;
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));

    int block = BLOCK_SIZE;
    int grid  = (n + block - 1) / block;
    int reps  = 200;

    auto bench = [&](auto fn) -> float {
        fn(); CUDA_CHECK(cudaDeviceSynchronize());
        GpuTimer t; t.start();
        for (int r = 0; r < reps; r++) fn();
        return t.stop_ms() / reps;
    };

    float ms1 = bench([&](){ smemStride1     <<<grid, block>>>(d_out, n); });
    float ms2 = bench([&](){ smemStride16    <<<grid, block>>>(d_out, n); });
    float ms3 = bench([&](){ smemStride32    <<<grid, block>>>(d_out, n); });
    float ms4 = bench([&](){ smemStride32Padded<<<grid, block>>>(d_out, n); });

    printf("Shared memory access pattern benchmark\n");
    printf("(all kernels do the same arithmetic — only access pattern differs)\n\n");
    printf("%-30s %10s %10s\n", "Kernel", "Time (µs)", "Relative");
    printf("%-30s %10s %10s\n", "──────────────────────────────", "─────────", "─────────");
    printf("%-30s %10.2f  1.00×  (no conflicts)\n",         "stride-1",             ms1 * 1000);
    printf("%-30s %10.2f  %.2f×  (2-way conflict)\n",       "stride-16",            ms2 * 1000, ms2/ms1);
    printf("%-30s %10.2f  %.2f×  (32-way conflict)\n",      "stride-32",            ms3 * 1000, ms3/ms1);
    printf("%-30s %10.2f  %.2f×  (padded, no conflict)\n",  "stride-32 (+1 pad)",  ms4 * 1000, ms4/ms1);

    printf("\n");
    printf("To confirm bank conflict counts with Nsight Compute:\n");
    printf("  ncu --metrics \\\n");
    printf("    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \\\n");
    printf("    ./build/03_bank_conflicts\n\n");
    printf("Expected: stride-1 → 0 conflicts, stride-32 → many conflicts,\n");
    printf("          stride-32 padded → 0 conflicts again.\n\n");
    printf("The +1 padding fix costs 128 bytes of shared memory per block.\n");
    printf("That is the cheapest performance win in GPU programming.\n");

    cudaFree(d_out);
    return 0;
}
