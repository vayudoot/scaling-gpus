// 02_transpose.cu — Post 3: GPU Memory: The Real Bottleneck
//
// Matrix transpose is the canonical example for teaching both coalescing
// and shared memory bank conflicts.
//
// Three kernels with increasing sophistication:
//
//   1. Naïve:
//      Reads are coalesced; writes scatter to column-strided addresses.
//      Write transactions: 32 per warp instead of 1.  Very slow.
//
//   2. Shared-memory-buffered (no padding):
//      Load into shared memory with coalesced reads.
//      Transpose in shared memory (free, on-chip).
//      Write from shared memory with coalesced writes.
//      BUT: the transposed read pattern causes 32-way shared memory bank
//      conflicts → serialised to 1/32 of shared memory bandwidth.
//
//   3. Shared-memory-buffered WITH +1 column padding:
//      Adding one extra column shifts every row's bank assignment by one.
//      All 32 threads in a warp now access different banks → no conflicts.
//      Both reads and writes to global memory are coalesced.
//      This is the correct implementation: full bandwidth.
//
// What to observe:
//   - Version 1 → Version 2: large speedup (fixing write coalescing)
//   - Version 2 → Version 3: further speedup (fixing bank conflicts)
//   - Version 3 bandwidth should approach the hardware ceiling

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <utility>
#include "../include/timer.cuh"

#define TILE 32   // tile dimension — 32 = one full warp per row/column

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 1: Naïve — coalesced reads, NON-COALESCED writes
// ─────────────────────────────────────────────────────────────────────────────
//
// Thread (row, col) reads in[row][col] and writes to out[col][row].
//
// Reads:  threads in a block row read consecutive in[row][col+0..31] → coalesced.
// Writes: threads write to out[col+0..31][row] — consecutive rows, same column.
//         That's a column of out[]: addresses are N floats apart.
//         HW issues 32 separate 128-byte transactions per warp write.

__global__ void transposeNaive(const float* __restrict__ in,
                                float*       __restrict__ out,
                                int N) {
    int col = blockIdx.x * TILE + threadIdx.x;
    int row = blockIdx.y * TILE + threadIdx.y;
    if (row < N && col < N)
        out[col * N + row] = in[row * N + col];
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 2: Shared-memory transpose WITHOUT padding — bank conflicts
// ─────────────────────────────────────────────────────────────────────────────
//
// Step 1: Each thread loads in[row][col] into tile[threadIdx.y][threadIdx.x].
//         Reads are coalesced (consecutive threadIdx.x → consecutive cols).
//         __syncthreads() ensures the tile is fully loaded.
//
// Step 2: Write tile[threadIdx.x][threadIdx.y] to coalesced out addresses.
//         The WRITE to global memory is now coalesced.
//
// PROBLEM: the read from shared memory in step 2:
//   tile[threadIdx.x][threadIdx.y]
//   Thread 0 reads tile[0][0] → bank 0
//   Thread 1 reads tile[1][0] → bank 0  (row 1, col 0 → address offset = 32*4 = 128 bytes = bank 0 again)
//   ...all 32 threads hit bank 0 → 32-way conflict → 32 serialised accesses.

__global__ void transposeSmemNoPad(const float* __restrict__ in,
                                    float*       __restrict__ out,
                                    int N) {
    __shared__ float tile[TILE][TILE];    // ← no padding, bank conflicts here

    int col_in = blockIdx.x * TILE + threadIdx.x;
    int row_in = blockIdx.y * TILE + threadIdx.y;

    if (row_in < N && col_in < N)
        tile[threadIdx.y][threadIdx.x] = in[row_in * N + col_in];  // coalesced read

    __syncthreads();

    int col_out = blockIdx.y * TILE + threadIdx.x;  // swapped block indices
    int row_out = blockIdx.x * TILE + threadIdx.y;

    if (row_out < N && col_out < N)
        out[row_out * N + col_out] = tile[threadIdx.x][threadIdx.y]; // ← 32-way bank conflict!
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 3: Shared-memory transpose WITH +1 padding — no conflicts
// ─────────────────────────────────────────────────────────────────────────────
//
// Change: __shared__ float tile[TILE][TILE + 1]
//
// Why it works:
//   Shared memory is organised as 32 banks, each 4 bytes wide.
//   Address byte offset k → bank (k / 4) % 32.
//
//   Without padding: tile[0][0] is at offset 0 (bank 0),
//                    tile[1][0] is at offset 32*4=128 bytes (bank 0 again).
//                    All row-0 column accesses go to bank 0 → conflict.
//
//   With +1 padding:  tile[0][0] is at offset 0 (bank 0),
//                     tile[1][0] is at offset (32+1)*4=132 bytes (bank 1).
//                     tile[2][0] is at offset (64+2)*4=264 bytes (bank 2).
//                     Every row starts in a different bank → no conflicts.
//
//   Cost: 32 extra floats per block (128 bytes) — negligible.

__global__ void transposeSmemPadded(const float* __restrict__ in,
                                     float*       __restrict__ out,
                                     int N) {
    __shared__ float tile[TILE][TILE + 1];  // ← +1 eliminates bank conflicts

    int col_in = blockIdx.x * TILE + threadIdx.x;
    int row_in = blockIdx.y * TILE + threadIdx.y;

    if (row_in < N && col_in < N)
        tile[threadIdx.y][threadIdx.x] = in[row_in * N + col_in];  // coalesced read

    __syncthreads();

    int col_out = blockIdx.y * TILE + threadIdx.x;
    int row_out = blockIdx.x * TILE + threadIdx.y;

    if (row_out < N && col_out < N)
        out[row_out * N + col_out] = tile[threadIdx.x][threadIdx.y]; // ← no conflict!
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static bool verify(const float* a, const float* b, int n) {
    for (int i = 0; i < n * n; i++) {
        if (fabsf(a[i] - b[i]) > 1e-5f) {
            fprintf(stderr, "Mismatch at %d: %.4f vs %.4f\n", i, a[i], b[i]);
            return false;
        }
    }
    return true;
}

template<typename Fn>
static float bench_ms(Fn fn, int reps = 20) {
    fn(); CUDA_CHECK(cudaDeviceSynchronize());
    GpuTimer t; t.start();
    for (int r = 0; r < reps; r++) fn();
    return t.stop_ms() / reps;
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    print_device_info();

    int N = (argc > 1) ? atoi(argv[1]) : 4096;
    // Round up to tile boundary so we don't need boundary checks in benchmarks
    N = ((N + TILE - 1) / TILE) * TILE;
    printf("Matrix : %d × %d  (%.1f MB)\n\n", N, N,
           (double)N * N * sizeof(float) / 1e6);

    size_t bytes = (size_t)N * N * sizeof(float);

    // Host allocations
    float* h_in  = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);
    float* h_ref = (float*)malloc(bytes);

    for (int i = 0; i < N * N; i++) h_in[i] = (float)i;

    // CPU reference transpose
    for (int r = 0; r < N; r++)
        for (int c = 0; c < N; c++)
            h_ref[c * N + r] = h_in[r * N + c];

    // Device allocations
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid(N / TILE, N / TILE);

    // Effective bandwidth: read N² floats + write N² floats = 2 × N² × 4 bytes
    size_t eff_bytes = 2ull * N * N * sizeof(float);

    printf("%-32s %10s %12s %10s\n",
           "Kernel", "Time (ms)", "BW (GB/s)", "Speedup");
    printf("%-32s %10s %12s %10s\n",
           "──────────────────────────────", "─────────", "──────────", "───────");

    float ms_naive;

    // Returns (ms, GB/s) — avoids structured bindings for broader compiler compat
    auto run_and_report = [&](const char* /*name*/, auto fn)
        -> std::pair<float,double> {
        CUDA_CHECK(cudaMemset(d_out, 0, bytes));
        float ms = bench_ms(fn);
        CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
        verify(h_out, h_ref, N);
        double bw = bandwidth_gb_s(eff_bytes, ms);
        return {ms, bw};
    };

    auto r1 = run_and_report("naïve", [&](){
        transposeNaive<<<grid, block>>>(d_in, d_out, N);
    });
    ms_naive = r1.first;
    printf("%-32s %10.3f %12.1f  1.00×\n", "Naive (coalesced R, strided W)",
           r1.first, r1.second);

    auto r2 = run_and_report("smem no padding", [&](){
        transposeSmemNoPad<<<grid, block>>>(d_in, d_out, N);
    });
    printf("%-32s %10.3f %12.1f  %.2f×\n", "Smem, no padding (bank conflicts)",
           r2.first, r2.second, ms_naive / r2.first);

    auto r3 = run_and_report("smem +1 padding", [&](){
        transposeSmemPadded<<<grid, block>>>(d_in, d_out, N);
    });
    printf("%-32s %10.3f %12.1f  %.2f×\n", "Smem, +1 padding (no conflicts)",
           r3.first, r3.second, ms_naive / r3.first);

    printf("\n");
    printf("Memory traffic explanation\n");
    printf("--------------------------\n");
    printf("Kernel 1 (naïve):\n");
    printf("  Reads : N² floats from consecutive row addresses     → 1 txn/warp\n");
    printf("  Writes: N² floats to column-strided addresses (N apart) → 32 txns/warp\n");
    printf("  Effective write waste: up to 32×\n\n");
    printf("Kernel 2 (smem, no pad):\n");
    printf("  HBM reads + writes: both coalesced ✓\n");
    printf("  Shared mem read: tile[threadIdx.x][threadIdx.y] → 32-way bank conflict ✗\n\n");
    printf("Kernel 3 (smem, +1 pad):\n");
    printf("  HBM reads + writes: both coalesced ✓\n");
    printf("  Shared mem read: every thread hits a different bank  ✓\n");
    printf("  Peak bandwidth utilisation: near hardware ceiling\n");

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out); free(h_ref);
    return 0;
}
