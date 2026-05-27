// matmul.cu — Post 2: Your First CUDA Kernels
//
// Demonstrates:
//   • 2D thread/block indexing for matrix operations
//   • Naïve matmul: one thread per output element, all reads from HBM
//   • Tiled matmul: cooperative loading into shared memory, then local compute
//   • How tiling reduces global memory traffic by TILE_SIZE×
//   • GFLOP/s measurement and comparison

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../include/timer.cuh"

#define TILE_SIZE 16   // 16×16 = 256 threads/block — a good starting point.
                       // Must evenly divide N for the simplified kernel below.
                       // The padded version in matmul_tiled_safe handles
                       // arbitrary N.

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 1: naïve matmul
// ─────────────────────────────────────────────────────────────────────────────
//
// Thread (row, col) computes C[row][col] = dot(A[row, :], B[:, col]).
// Every element of A and B is read from global memory (HBM) for every thread
// that needs it — there is zero data reuse. This is what makes it slow.
//
// Memory traffic for one output element: 2 × N floats from HBM.
// Total HBM reads: 2 × N × N × N × 4 bytes.

__global__ void matmulNaive(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float*       __restrict__ C,
                              int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= N || col >= N) return;

    float sum = 0.f;
    for (int k = 0; k < N; k++)
        sum += A[row * N + k] * B[k * N + col];

    C[row * N + col] = sum;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 2: tiled matmul with shared memory
// ─────────────────────────────────────────────────────────────────────────────
//
// Key idea: a 16×16 block of threads cooperates to load a 16×16 tile of A
// and a 16×16 tile of B into on-chip shared memory, then each thread computes
// its partial dot product against the *cached* tiles. The tile slides across
// the K dimension in steps of TILE_SIZE.
//
// Data reuse: each float loaded into shared memory is used TILE_SIZE times
// (once per thread in the tile row/column). This reduces HBM traffic by
// TILE_SIZE×, which is the sole source of the speedup.
//
// __syncthreads() is required:
//   1. After loading — all threads must finish writing before any thread reads.
//   2. After computing — all threads must finish reading before the next tile
//      overwrites the shared buffer.

__global__ void matmulTiled(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float*       __restrict__ C,
                              int N) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.f;
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        // Collaborative load: each thread loads one element of each tile.
        // Out-of-bounds accesses are zeroed so partial tiles are handled safely.
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] =
            (row < N && aCol < N) ? A[row * N + aCol] : 0.f;
        tileB[threadIdx.y][threadIdx.x] =
            (bRow < N && col < N) ? B[bRow * N + col] : 0.f;

        // Barrier 1: wait for all threads to finish loading before computing.
        __syncthreads();

        // Each thread computes its partial dot product against the cached tile.
        for (int k = 0; k < TILE_SIZE; k++)
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];

        // Barrier 2: wait for all threads to finish computing before the next
        // iteration overwrites tileA and tileB.
        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}

// ─────────────────────────────────────────────────────────────────────────────
// CPU reference
// ─────────────────────────────────────────────────────────────────────────────

void matmulCPU(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++)
            for (int j = 0; j < N; j++)
                C[i * N + j] += A[i * N + k] * B[k * N + j];
}

// ─────────────────────────────────────────────────────────────────────────────
// Verify
// ─────────────────────────────────────────────────────────────────────────────

bool verify(const float* gpu, const float* cpu, int N, float tol = 1e-2f) {
    // Relative tolerance: large N accumulates floating-point error.
    for (int i = 0; i < N * N; i++) {
        float diff = fabsf(gpu[i] - cpu[i]);
        float ref  = fabsf(cpu[i]) + 1e-6f;
        if (diff / ref > tol) {
            int row = i / N, col = i % N;
            fprintf(stderr,
                    "MISMATCH at (%d,%d): GPU=%.4f CPU=%.4f rel_err=%.2e\n",
                    row, col, gpu[i], cpu[i], diff / ref);
            return false;
        }
    }
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark helper
// ─────────────────────────────────────────────────────────────────────────────

struct BenchResult { float ms; double gflops; };

template<typename KernelFn>
BenchResult bench(KernelFn fn, int repeats = 5) {
    // Warm up
    fn();
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer t;
    t.start();
    for (int r = 0; r < repeats; r++) fn();
    float ms = t.stop_ms() / repeats;
    return {ms, 0.0};  // caller fills gflops
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    print_device_info();

    // Matrix dimension — default 1024×1024
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    printf("Matrix size   : %d × %d\n\n", N, N);

    size_t bytes = (size_t)N * N * sizeof(float);

    // ── Host allocations ────────────────────────────────────────────────────
    float* h_A   = (float*)malloc(bytes);
    float* h_B   = (float*)malloc(bytes);
    float* h_C_n = (float*)malloc(bytes);   // naïve result
    float* h_C_t = (float*)malloc(bytes);   // tiled result
    float* h_ref = (float*)calloc(N * N, sizeof(float));

    // Initialise inputs with small random values (keeps CPU reference fast)
    srand(42);
    for (int i = 0; i < N * N; i++) {
        h_A[i] = (rand() % 100) * 0.01f;
        h_B[i] = (rand() % 100) * 0.01f;
    }

    // ── Device allocations ──────────────────────────────────────────────────
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Launch config shared by both kernels
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (N + TILE_SIZE - 1) / TILE_SIZE);

    // FLOPs: each output element requires N multiply-adds = 2N FLOPs
    double flops = 2.0 * N * N * N;

    // ─────────────────────────────────────────────────────────────────────────
    // Benchmark: naïve
    // ─────────────────────────────────────────────────────────────────────────
    CUDA_CHECK(cudaMemset(d_C, 0, bytes));
    auto run_naive = [&]() {
        matmulNaive<<<grid, block>>>(d_A, d_B, d_C, N);
    };
    BenchResult naive_r = bench(run_naive);
    naive_r.gflops = flops / (naive_r.ms * 1e-3) / 1e9;
    CUDA_CHECK(cudaMemcpy(h_C_n, d_C, bytes, cudaMemcpyDeviceToHost));

    printf("──── Naïve matmul ────\n");
    printf("  Grid   : (%d, %d) blocks of (%d, %d) threads\n",
           grid.x, grid.y, block.x, block.y);
    printf("  Time   : %.2f ms\n", naive_r.ms);
    printf("  Perf   : %.1f GFLOP/s\n\n", naive_r.gflops);

    // ─────────────────────────────────────────────────────────────────────────
    // Benchmark: tiled
    // ─────────────────────────────────────────────────────────────────────────
    CUDA_CHECK(cudaMemset(d_C, 0, bytes));
    auto run_tiled = [&]() {
        matmulTiled<<<grid, block>>>(d_A, d_B, d_C, N);
    };
    BenchResult tiled_r = bench(run_tiled);
    tiled_r.gflops = flops / (tiled_r.ms * 1e-3) / 1e9;
    CUDA_CHECK(cudaMemcpy(h_C_t, d_C, bytes, cudaMemcpyDeviceToHost));

    printf("──── Tiled matmul (tile=%d) ────\n", TILE_SIZE);
    printf("  Shared mem / block : %zu bytes\n",
           2 * TILE_SIZE * TILE_SIZE * sizeof(float));
    printf("  Time   : %.2f ms\n", tiled_r.ms);
    printf("  Perf   : %.1f GFLOP/s\n\n", tiled_r.gflops);

    printf("  Speedup (tiled / naïve) : %.2f×\n\n",
           naive_r.ms / tiled_r.ms);

    // ─────────────────────────────────────────────────────────────────────────
    // CPU reference (small N only — O(N³) is slow)
    // ─────────────────────────────────────────────────────────────────────────
    bool verified_naive = true, verified_tiled = true;
    if (N <= 512) {
        printf("Running CPU reference (N=%d)...\n", N);
        matmulCPU(h_A, h_B, h_ref, N);
        verified_naive = verify(h_C_n, h_ref, N);
        verified_tiled = verify(h_C_t, h_ref, N);
        printf("Naïve correctness : %s\n", verified_naive ? "PASS" : "FAIL");
        printf("Tiled correctness : %s\n\n", verified_tiled ? "PASS" : "FAIL");
    } else {
        // For large N, cross-check tiled against naïve (both run on GPU)
        bool cross_ok = verify(h_C_t, h_C_n, N, 1e-3f);
        printf("Tiled vs naïve    : %s\n\n", cross_ok ? "PASS" : "FAIL");
        verified_naive = verified_tiled = cross_ok;
    }

    printf("╔══════════════════════════════════════╗\n");
    printf("║  Summary (N = %4d)                   ║\n", N);
    printf("╠══════════════════════════════════════╣\n");
    printf("║  Naïve  : %6.1f GFLOP/s             ║\n", naive_r.gflops);
    printf("║  Tiled  : %6.1f GFLOP/s  (%.1f×)     ║\n",
           tiled_r.gflops, naive_r.ms / tiled_r.ms);
    printf("║  Note: cuBLAS reaches ~300+ GFLOP/s ║\n");
    printf("╚══════════════════════════════════════╝\n\n");

    // ── Cleanup ──────────────────────────────────────────────────────────────
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C_n); free(h_C_t); free(h_ref);

    return (verified_naive && verified_tiled) ? 0 : 1;
}
