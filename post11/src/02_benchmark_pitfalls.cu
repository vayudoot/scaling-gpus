// 02_benchmark_pitfalls.cu  --  Post 11: Profiling and Optimization
//
// "Measure, don't guess" -- but measuring wrong is worse than guessing,
// because it comes with false confidence. This program commits the four
// classic benchmarking sins on the SAME workload (one cuBLAS GEMM) and
// prints what each would have made you believe, next to the truth.
//
//   SIN 1: timing the first call         (JIT, cuBLAS heuristics, clocks)
//   SIN 2: CPU timer without a sync      (you time the LAUNCH, not the work)
//   SIN 3: reporting a single run        (variance, clock drift)
//   SIN 4: including the H2D copy        (measuring PCIe, calling it compute)
//
// The right way, used everywhere in this series:
//   warm up  ->  cudaEvent pair around the region  ->  median of many reps

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <chrono>
#include <vector>
#include "../include/utils.cuh"

int main() {
    print_device_info();
    srand(42);

    const int N = 2048;                       // GEMM size N x N x N
    const double flops = 2.0 * N * (double)N * N;

    cublasHandle_t cublas; CUBLAS_CHECK(cublasCreate(&cublas));
    float one = 1.f, zero = 0.f;

    size_t bytes = (size_t)N * N * sizeof(float);
    float* h_A = (float*)malloc(bytes);
    rand_fill(h_A, N * N, -1.f, 1.f);
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_A, bytes, cudaMemcpyHostToDevice));

    printf("Workload: cublasSgemm %d x %d x %d  (%.1f GFLOP)\n", N, N, N, flops/1e9);

    // ── SIN 1: time the very first call ──────────────────────────────────────
    section("Sin 1: timing the first call");
    float ms_first;
    {
        GpuTimer t; t.start();
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            N, N, N, &one, d_A, N, d_B, N, &zero, d_C, N));
        ms_first = t.stop_ms();
    }
    printf("  first call        : %8.3f ms  (%6.1f GFLOP/s)\n",
           ms_first, flops / (ms_first * 1e-3) / 1e9);
    printf("  includes cuBLAS heuristic selection, module load, cold clocks.\n");

    // ── the honest baseline: warm up, then median of 50 event-timed reps ─────
    for (int w = 0; w < 5; w++)
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            N, N, N, &one, d_A, N, d_B, N, &zero, d_C, N));
    CUDA_CHECK(cudaDeviceSynchronize());

    const int REPS = 50;
    std::vector<float> samples(REPS);
    for (int r = 0; r < REPS; r++) {
        GpuTimer t; t.start();
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            N, N, N, &one, d_A, N, d_B, N, &zero, d_C, N));
        samples[r] = t.stop_ms();
    }
    std::sort(samples.begin(), samples.end());
    float ms_min = samples[0], ms_med = samples[REPS/2], ms_max = samples[REPS-1];
    printf("\n  after warmup, %d event-timed reps:\n", REPS);
    printf("  min / median / max: %.3f / %.3f / %.3f ms\n", ms_min, ms_med, ms_max);
    printf("  first call was %.1fx slower than the steady state.\n",
           ms_first / ms_med);

    // ── SIN 2: CPU timer without synchronization ──────────────────────────────
    section("Sin 2: CPU wall clock without a sync");
    {
        auto c0 = std::chrono::high_resolution_clock::now();
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            N, N, N, &one, d_A, N, d_B, N, &zero, d_C, N));
        auto c1 = std::chrono::high_resolution_clock::now();   // NO sync!
        double ms_nosync = std::chrono::duration<double, std::milli>(c1-c0).count();

        auto c2 = std::chrono::high_resolution_clock::now();
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            N, N, N, &one, d_A, N, d_B, N, &zero, d_C, N));
        CUDA_CHECK(cudaDeviceSynchronize());                   // sync
        auto c3 = std::chrono::high_resolution_clock::now();
        double ms_sync = std::chrono::duration<double, std::milli>(c3-c2).count();

        printf("  chrono, no sync   : %8.3f ms  -> \"%.0f TFLOP/s\"!  (a lie:\n",
               ms_nosync, flops / (ms_nosync * 1e-3) / 1e12);
        printf("                      kernel launch is ASYNC; you timed the enqueue)\n");
        printf("  chrono, with sync : %8.3f ms  (close to the event timing)\n", ms_sync);
        printf("  cudaEvent median  : %8.3f ms  (the number to report)\n", ms_med);
    }

    // ── SIN 3: single-run reporting ───────────────────────────────────────────
    section("Sin 3: reporting one run");
    printf("  spread across %d reps: %.3f .. %.3f ms (%.1f%% max-over-min)\n",
           REPS, ms_min, ms_max, 100.0 * (ms_max/ms_min - 1.0));
    printf("  A single unlucky sample misreports throughput by that margin.\n");
    printf("  Report the median (robust) or min (best-case, note it as such);\n");
    printf("  lock clocks for papers: nvidia-smi -lgc <min,max>.\n");

    // ── SIN 4: including the transfer ─────────────────────────────────────────
    section("Sin 4: timing H2D + kernel and calling it \"kernel time\"");
    {
        GpuTimer t; t.start();
        CUDA_CHECK(cudaMemcpy(d_B, h_A, bytes, cudaMemcpyHostToDevice));
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            N, N, N, &one, d_A, N, d_B, N, &zero, d_C, N));
        float ms_with_copy = t.stop_ms();
        printf("  H2D + GEMM        : %8.3f ms  -> \"%6.1f GFLOP/s\"\n",
               ms_with_copy, flops / (ms_with_copy * 1e-3) / 1e9);
        printf("  GEMM alone        : %8.3f ms  ->  %6.1f GFLOP/s\n",
               ms_med, flops / (ms_med * 1e-3) / 1e9);
        printf("  The copy is real cost -- but it is PCIe cost. Report it as\n");
        printf("  its own line, or hide it with async copies (Post 4).\n");
    }

    section("The recipe used throughout this series");
    printf("  1. warm up (>= 3 calls), cudaDeviceSynchronize\n");
    printf("  2. time with cudaEvent pairs around the DEVICE work only\n");
    printf("  3. repeat >= 20x, report the median; note clocks if publishing\n");
    printf("  4. verify the output every time you change the code being timed\n");

    cublasDestroy(cublas);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); free(h_A);
    return 0;
}
