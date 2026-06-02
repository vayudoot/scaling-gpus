// 02_tensor_cores.cu  --  Post 7: Mixed Precision and Quantization
//
// Tensor Cores perform a 16x16x16 matrix multiply-accumulate (MMA) in one
// instruction using the warp-level WMMA API (nvcuda::wmma).
//
// This program:
//   A. Implements a matmul using raw WMMA intrinsics -- showing exactly what
//      Tensor Cores are doing at the hardware level.
//   B. Benchmarks WMMA vs cuBLAS FP16 vs cuBLAS FP32 to place WMMA in context.
//   C. Demonstrates the layout failure mode: what happens when you accidentally
//      pass FP32 data to a kernel expecting FP16 (silent wrong results at
//      FP32 speed, not a crash).
//   D. Shows how to check Tensor Core utilization via Nsight Compute.
//
// WMMA programming model:
//   - Operates at WARP level (all 32 threads collaborate)
//   - Inputs: 16x16 matrix fragments distributed across warp registers
//   - One mma_sync() call = 16x16x16 fused multiply-add = 8192 FLOPs
//   - Accumulator held in FP32 (even though inputs are FP16/BF16)
//
// Key layout requirement:
//   Fragment A: row_major (each row is contiguous in memory)
//   Fragment B: col_major (each column is contiguous in memory)
//   Swapping these compiles silently but produces wrong results.
//
// In practice you rarely write WMMA kernels directly -- cuBLAS, cuDNN, and
// libraries like CUTLASS handle this. Writing one manually builds the intuition
// for why dimension alignment and data types matter so much.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>       // nvcuda::wmma -- Tensor Core WMMA API
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/utils.cuh"

using namespace nvcuda;

// WMMA tile dimensions -- must be 16 for the basic MMA
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// ─────────────────────────────────────────────────────────────────────────────
// WMMA matmul kernel: C[M,N] = A[M,K] * B[K,N]
//
// Grid:  (M/WMMA_M, N/WMMA_N) warps -- one warp per 16x16 output tile
// Block: warpSize (32) threads per warp * warps_per_block
//
// Each warp:
//   1. Loop over K in WMMA_K=16 steps
//   2. load_matrix_sync A tile [16x16] into fragA (row_major)
//   3. load_matrix_sync B tile [16x16] into fragB (col_major)
//   4. mma_sync: fragC += fragA * fragB  (one instruction, 8192 FLOPs)
//   5. store_matrix_sync: write fragC [16x16, FP32] to global memory
// ─────────────────────────────────────────────────────────────────────────────
__global__ void wmmaMatmul(const __half* __restrict__ A,   // [M x K] row-major
                             const __half* __restrict__ B,   // [K x N] col-major
                             float*        __restrict__ C,   // [M x N] row-major output
                             int M, int N, int K) {
    // Each block contains multiple warps. Identify which warp this is.
    int warp_id = threadIdx.x / warpSize;
    int warps_per_block_x = blockDim.x / warpSize;   // warps in x direction

    // Which 16x16 tile does this warp compute?
    int warp_row = blockIdx.y * (blockDim.y) + threadIdx.y;  // tile row index
    int warp_col = blockIdx.x * warps_per_block_x + warp_id; // tile col index

    if (warp_row * WMMA_M >= M || warp_col * WMMA_N >= N) return;

    // Declare fragments
    // fragA: 16x16 slice of A, FP16, row_major
    // fragB: 16x16 slice of B, FP16, col_major
    // fragC: 16x16 accumulator, FP32 (accumulates in higher precision)
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                   __half, wmma::row_major> fragA;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                   __half, wmma::col_major> fragB;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> fragC;

    // Zero the accumulator
    wmma::fill_fragment(fragC, 0.0f);

    // Sweep the K dimension in WMMA_K=16 steps
    for (int k = 0; k < K; k += WMMA_K) {
        int aRow = warp_row * WMMA_M;
        int aCol = k;
        int bRow = k;
        int bCol = warp_col * WMMA_N;

        if (aRow < M && aCol < K && bRow < K && bCol < N) {
            // Load a 16x16 tile of A from global memory into fragA.
            // The third argument is the leading dimension (stride between rows).
            wmma::load_matrix_sync(fragA, A + aRow * K + aCol, K);

            // Load a 16x16 tile of B (col_major: stride = K, not N).
            // B is stored as [K x N] but we access it col-major for the MMA.
            wmma::load_matrix_sync(fragB, B + bRow + bCol * K, K);

            // Tensor Core MMA: fragC += fragA * fragB
            // This single instruction does 16*16*16*2 = 8192 FLOPs.
            wmma::mma_sync(fragC, fragA, fragB, fragC);
        }
    }

    // Write the 16x16 FP32 result tile back to global memory
    int cRow = warp_row * WMMA_M;
    int cCol = warp_col * WMMA_N;
    if (cRow < M && cCol < N) {
        wmma::store_matrix_sync(C + cRow * N + cCol, fragC, N, wmma::mem_row_major);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Wrong layout kernel: shows what happens when B layout is wrong
// This silently produces wrong results -- not a crash, not NaN.
// This is the most dangerous silent failure mode in Tensor Core programming.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void wmmaMatmulWrongLayout(const __half* __restrict__ A,
                                       const __half* __restrict__ B,
                                       float*        __restrict__ C,
                                       int M, int N, int K) {
    int warp_id   = threadIdx.x / warpSize;
    int warp_row  = blockIdx.y;
    int warp_col  = blockIdx.x * (blockDim.x / warpSize) + warp_id;

    if (warp_row * WMMA_M >= M || warp_col * WMMA_N >= N) return;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                   __half, wmma::row_major> fragA;
    // BUG: using col_major for fragment_b when B is actually row_major in memory
    // The correct layout for B stored [K x N] row-major is col_major.
    // Using row_major here reads B transposed -> wrong result, no error.
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                   __half, wmma::row_major> fragB;  // <- wrong! should be col_major
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> fragC;
    wmma::fill_fragment(fragC, 0.0f);

    for (int k = 0; k < K; k += WMMA_K) {
        int aRow = warp_row * WMMA_M, aCol = k;
        int bRow = k, bCol = warp_col * WMMA_N;
        if (aRow < M && aCol < K && bRow < K && bCol < N) {
            wmma::load_matrix_sync(fragA, A + aRow * K + aCol, K);
            wmma::load_matrix_sync(fragB, B + bRow * N + bCol, N); // using row stride
            wmma::mma_sync(fragC, fragA, fragB, fragC);
        }
    }
    int cRow = warp_row * WMMA_M, cCol = warp_col * WMMA_N;
    if (cRow < M && cCol < N)
        wmma::store_matrix_sync(C + cRow * N + cCol, fragC, N, wmma::mem_row_major);
}

// ─────────────────────────────────────────────────────────────────────────────
// CPU reference matmul for correctness check
// ─────────────────────────────────────────────────────────────────────────────
static void cpuMatmul(const float* A, const float* B, float* C,
                       int M, int N, int K) {
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float s = 0.f;
            for (int k = 0; k < K; k++) s += A[m*K+k] * B[k*N+n];
            C[m*N+n] = s;
        }
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main() {
    print_device_info();
    srand(42);

    // WMMA requires sm_70+ (Volta or newer)
    int dev; CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    if (prop.major < 7) {
        printf("WMMA requires sm_70+ (Volta or newer). Exiting.\n");
        return 0;
    }

    // ── A: Correctness check on small matrix ─────────────────────────────────
    section("A: WMMA correctness check (M=N=K=32)");
    {
        const int M = 32, N = 32, K = 32;
        float* h_A = (float*)malloc(M*K*sizeof(float));
        float* h_B = (float*)malloc(K*N*sizeof(float));
        float* h_C_ref = (float*)malloc(M*N*sizeof(float));
        float* h_C_gpu = (float*)malloc(M*N*sizeof(float));

        rand_fill_fp32(h_A, M*K, -0.5f, 0.5f);
        rand_fill_fp32(h_B, K*N, -0.5f, 0.5f);
        cpuMatmul(h_A, h_B, h_C_ref, M, N, K);

        // Convert inputs to FP16 for WMMA
        __half* h_A16 = (__half*)malloc(M*K*sizeof(__half));
        __half* h_B16 = (__half*)malloc(K*N*sizeof(__half));
        // B must be stored col-major for the wmma::col_major fragment
        __half* h_B16_col = (__half*)malloc(K*N*sizeof(__half));
        fp32_to_fp16(h_A, h_A16, M*K);
        fp32_to_fp16(h_B, h_B16, K*N);
        // Transpose B to col-major: B_col[k][n] = B_row[k][n] but stored as B_col[n][k]
        // Actually col_major means column stride = 1, so B[k,n] is at B[n*K + k]
        for (int k = 0; k < K; k++)
            for (int n = 0; n < N; n++)
                h_B16_col[n * K + k] = h_B16[k * N + n];

        __half *d_A16, *d_B16_col;
        float  *d_C;
        CUDA_CHECK(cudaMalloc(&d_A16,     M*K*sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_B16_col, K*N*sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_C,       M*N*sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_A16,     h_A16,     M*K*sizeof(__half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B16_col, h_B16_col, K*N*sizeof(__half), cudaMemcpyHostToDevice));

        // warpSize=32 is a device built-in; use the constant 32 in host code.
        // 1 warp per 16x16 output tile: block=(32,1), grid=(N_tiles_x, N_tiles_y)
        const int WARP = 32;
        dim3 blk2(WARP, 1);
        dim3 grd2(N / WMMA_N, M / WMMA_M);

        wmmaMatmul<<<grd2, blk2>>>(d_A16, d_B16_col, d_C, M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost));

        // Compare -- FP16 input means small rounding errors are expected
        float err = max_abs_diff_fp32(h_C_gpu, h_C_ref, M*N);
        printf("  Max error vs FP32 CPU reference: %.4e  %s\n\n", err,
               err < 0.1f ? "PASS" : "FAIL");
        printf("  Note: small errors (~0.01) are expected because FP16 inputs\n");
        printf("  have less precision than the FP32 CPU reference.\n");

        // Wrong layout demo
        CUDA_CHECK(cudaMemset(d_C, 0, M*N*sizeof(float)));
        wmmaMatmulWrongLayout<<<grd2, blk2>>>(d_A16, d_A16, d_C, M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost));
        // This result is wrong -- but no error was raised
        printf("  Wrong layout kernel ran without error. First output: %.4f\n",
               h_C_gpu[0]);
        printf("  This is a silently wrong result -- no crash, no NaN.\n");
        printf("  Always verify WMMA results against a reference implementation.\n");

        cudaFree(d_A16); cudaFree(d_B16_col); cudaFree(d_C);
        free(h_A); free(h_B); free(h_C_ref); free(h_C_gpu);
        free(h_A16); free(h_B16); free(h_B16_col);
    }

    // ── B: Performance comparison ─────────────────────────────────────────────
    section("B: Throughput -- WMMA vs cuBLAS FP16 vs cuBLAS FP32");
    {
        const int M = 4096, N = 4096, K = 4096;
        double flops = 2.0 * M * N * K;

        cublasHandle_t handle;
        CUBLAS_CHECK(cublasCreate(&handle));

        // Allocate matrices
        float  *d_A32, *d_B32, *d_C32;
        __half *d_A16, *d_B16, *d_B16_col, *d_C16;
        CUDA_CHECK(cudaMalloc(&d_A32, (size_t)M*K*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B32, (size_t)K*N*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C32, (size_t)M*N*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_A16, (size_t)M*K*sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_B16, (size_t)K*N*sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_B16_col, (size_t)K*N*sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_C16, (size_t)M*N*sizeof(__half)));
        CUDA_CHECK(cudaMemset(d_A32, 1, (size_t)M*K*sizeof(float)));
        CUDA_CHECK(cudaMemset(d_B32, 1, (size_t)K*N*sizeof(float)));
        CUDA_CHECK(cudaMemset(d_A16, 1, (size_t)M*K*sizeof(__half)));
        CUDA_CHECK(cudaMemset(d_B16, 1, (size_t)K*N*sizeof(__half)));
        CUDA_CHECK(cudaMemset(d_B16_col, 1, (size_t)K*N*sizeof(__half)));

        const int REPS = 20;

        float alpha32 = 1.f, beta32 = 0.f;
        __half alpha16 = __float2half(1.f), beta16 = __float2half(0.f);

        // cuBLAS FP32
        float ms_fp32;
        {
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K, &alpha32, d_B32, N, d_A32, K, &beta32, d_C32, N);
            CUDA_CHECK(cudaDeviceSynchronize());
            GpuTimer t; t.start();
            for (int r = 0; r < REPS; r++)
                cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha32, d_B32, N, d_A32, K, &beta32, d_C32, N);
            ms_fp32 = t.stop_ms() / REPS;
        }

        // cuBLAS FP16 Tensor Cores (cublasHgemm)
        float ms_fp16_cublas;
        {
            cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K, &alpha16, d_B16, N, d_A16, K, &beta16, d_C16, N);
            CUDA_CHECK(cudaDeviceSynchronize());
            GpuTimer t; t.start();
            for (int r = 0; r < REPS; r++)
                cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha16, d_B16, N, d_A16, K, &beta16, d_C16, N);
            ms_fp16_cublas = t.stop_ms() / REPS;
        }

        // Our WMMA kernel
        dim3 blk(32, 1);   // one warp = 32 threads
        dim3 grd_wmma(N / WMMA_N, M / WMMA_M);
        float ms_wmma;
        {
            wmmaMatmul<<<grd_wmma, blk>>>(d_A16, d_B16_col, d_C32, M, N, K);
            CUDA_CHECK(cudaDeviceSynchronize());
            GpuTimer t; t.start();
            for (int r = 0; r < REPS; r++)
                wmmaMatmul<<<grd_wmma, blk>>>(d_A16, d_B16_col, d_C32, M, N, K);
            ms_wmma = t.stop_ms() / REPS;
        }

        printf("  Matrix %dx%dx%d  (%.0f GFLOP)\n\n", M, N, K, flops/1e9);
        printf("  %-30s %10s %12s %10s\n",
               "Method", "Time(ms)", "GFLOP/s", "vs FP32");
        printf("  %-30s %10s %12s %10s\n",
               "------------------------------","--------","----------","--------");

        double g32 = gflops(flops, ms_fp32);
        double g16 = gflops(flops, ms_fp16_cublas);
        double gw  = gflops(flops, ms_wmma);

        printf("  %-30s %10.2f %12.1f  1.00x\n",
               "cuBLAS FP32 (CUDA cores)", ms_fp32, g32);
        printf("  %-30s %10.2f %12.1f  %.1fx\n",
               "Our WMMA (Tensor Cores)", ms_wmma, gw, g32/ms_wmma*ms_fp32);
        printf("  %-30s %10.2f %12.1f  %.1fx\n",
               "cuBLAS FP16 (Tensor Cores)", ms_fp16_cublas, g16, g32/ms_fp16_cublas*ms_fp32);

        printf("\n  cuBLAS vs our WMMA gap: cuBLAS uses register-level tiling,\n");
        printf("  pipelined memory loads (cp.async), and hand-tuned inner loops.\n");
        printf("  Our WMMA kernel shows the programming model; production code\n");
        printf("  should always use cuBLAS or CUTLASS for dense matmuls.\n\n");
        printf("  The Tensor Core speedup vs CUDA cores is the 15x number from\n");
        printf("  the post. Every LLM forward pass depends on this gap.\n");

        cublasDestroy(handle);
        cudaFree(d_A32); cudaFree(d_B32); cudaFree(d_C32);
        cudaFree(d_A16); cudaFree(d_B16); cudaFree(d_B16_col); cudaFree(d_C16);
    }

    section("How to verify Tensor Core utilization");
    printf("  Run Nsight Compute:\n");
    printf("    ncu --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active\n");
    printf("        ./build/02_tensor_cores\n\n");
    printf("  For cuBLAS FP16: expect >50%% tensor pipe utilization.\n");
    printf("  For cuBLAS FP32: expect ~0%% (CUDA cores, not Tensor Cores).\n");
    printf("  If tensor utilization is unexpectedly low, check:\n");
    printf("    1. Input dtype is __half or __nv_bfloat16, not float\n");
    printf("    2. Matrix dimensions are multiples of 16 (ideally 64 or 128)\n");
    printf("    3. Memory is properly aligned (256-byte alignment optimal)\n");

    return 0;
}
