// 01_precision_formats.cu  --  Post 7: Mixed Precision and Quantization
//
// Demonstrates the practical differences between FP32, FP16, and BF16:
//
//   A. Overflow demo: FP16 overflows at ~65504. BF16 has the same range as
//      FP32 (~3.4e38) because it keeps FP32's 8-bit exponent. This is the
//      primary reason BF16 replaced FP16 for training.
//
//   B. Throughput benchmark: elementwise kernel throughput at each precision.
//      FP16 and BF16 both use 2 bytes/element; FP32 uses 4 bytes/element.
//      For memory-bound kernels, FP16/BF16 achieve ~2x the throughput of FP32.
//
//   C. Matmul throughput: FP32 on CUDA cores vs FP16 on Tensor Cores.
//      This is the 15x gap from the post — the most important number.
//
//   D. Precision loss: shows how BF16's 7-bit mantissa (vs FP32's 23-bit)
//      causes rounding for small gradient updates — why optimizer state must
//      stay in FP32.
//
// Number format reference:
//   FP32:  1 sign + 8 exp + 23 mantissa = 32 bits  max ~3.4e38
//   BF16:  1 sign + 8 exp +  7 mantissa = 16 bits  max ~3.4e38  (same range!)
//   FP16:  1 sign + 5 exp + 10 mantissa = 16 bits  max ~6.5e4   (narrow range!)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/utils.cuh"

// ─────────────────────────────────────────────────────────────────────────────
// Elementwise kernels at different precisions
// ─────────────────────────────────────────────────────────────────────────────

__global__ void scaleKernelFP32(const float* __restrict__ in,
                                  float*       __restrict__ out,
                                  float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * s + (1.f - s);
}

__global__ void scaleKernelFP16(const __half* __restrict__ in,
                                  __half*       __restrict__ out,
                                  __half s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __hadd(__hmul(in[i], s),
                                __hmul(__float2half(1.f), __hsub(__float2half(1.f), s)));
}


// ─────────────────────────────────────────────────────────────────────────────
// Overflow demonstration kernel
// ─────────────────────────────────────────────────────────────────────────────
// Show that summing a large FP16 value produces infinity, but the same value
// in BF16 is fine because BF16 has 8 exponent bits (same as FP32).

__global__ void overflowDemoKernel(float fp32_val, __half* fp16_result,
                                    float* fp32_result) {
    // fp16_result will hold the result of squaring fp32_val in FP16
    __half h = __float2half(fp32_val);
    fp16_result[0] = __hmul(h, h);           // likely infinity if fp32_val is large
    fp32_result[0] = fp32_val * fp32_val;    // fine in FP32
}

// ─────────────────────────────────────────────────────────────────────────────
// Precision loss demo: accumulating small updates in FP32 vs BF16
// ─────────────────────────────────────────────────────────────────────────────
// In training, gradient updates are often very small (e.g., 1e-4 * large_param).
// BF16's 7-bit mantissa means values differing by less than ~2^-7 relative to
// the larger one get rounded away — the small update is lost entirely.
// This is why optimizer state must remain in FP32.

__global__ void precisionLossDemo(float* results) {
    // Accumulate 1000 small increments onto a large base value
    // FP32: has 23 mantissa bits, can distinguish 1e3 / 1e6 = 1e-3 relative diff
    // BF16: has 7 mantissa bits, can distinguish ~1/128 = 0.78% relative diff
    //       smaller updates are simply rounded to zero

    float  base_fp32 = 1000.f;
    float  delta     = 0.001f;   // 0.001% of base_fp32

    // FP32 accumulation — should give 1001.0
    float acc_fp32 = base_fp32;
    for (int i = 0; i < 1000; i++) acc_fp32 += delta;
    results[0] = acc_fp32;

    // BF16 simulation: round base to BF16 precision, then accumulate
    // BF16 has ~2 decimal digits of precision relative to the number's magnitude.
    // base=1000: smallest representable delta ≈ 1000 * 2^-7 ≈ 7.8
    // delta=0.001 is far below this threshold — it rounds to zero every time.
    __nv_bfloat16 acc_bf16 = __float2bfloat16(base_fp32);
    __nv_bfloat16 d_bf16   = __float2bfloat16(delta);
    for (int i = 0; i < 1000; i++) acc_bf16 = __hadd(acc_bf16, d_bf16);
    results[1] = __bfloat162float(acc_bf16);

    results[2] = base_fp32 + 1000.f * delta;  // expected: 1001.0
}

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark helper
// ─────────────────────────────────────────────────────────────────────────────
template<typename Fn>
static float bench(Fn fn, int reps = 100) {
    fn(); CUDA_CHECK(cudaDeviceSynchronize());
    GpuTimer t; t.start();
    for (int r = 0; r < reps; r++) fn();
    return t.stop_ms() / reps;
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main() {
    print_device_info();
    srand(42);

    // Check if BF16 is supported (requires sm_80+)
    int dev; CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    bool has_bf16 = (prop.major >= 8);

    // ── A: Overflow demo ─────────────────────────────────────────────────────
    section("A: FP16 overflow vs BF16/FP32 range");
    {
        // FP16 max is ~65504. Values beyond this become infinity.
        float test_vals[] = {100.f, 255.f, 32767.f, 65504.f, 65505.f, 1e6f};
        printf("  %-12s %-15s %-15s %-10s\n",
               "Value", "FP16 result", "FP32 result", "FP16 OK?");
        printf("  %-12s %-15s %-15s %-10s\n",
               "------------","---------------","---------------","--------");

        for (float v : test_vals) {
            __half h = __float2half(v);
            float back = __half2float(h);
            printf("  %-12g %-15g %-15g %-10s\n",
                   (double)v, (double)back, (double)v,
                   (isinf(back) || fabsf(back - v) / (fabsf(v) + 1e-9f) > 0.01f)
                   ? "OVERFLOW/LOSS" : "OK");
        }

        printf("\n  BF16 uses 8 exponent bits (same as FP32): max ~3.4e38\n");
        printf("  FP16 uses 5 exponent bits: max ~65504\n");
        printf("  During training, logits and pre-softmax activations can\n");
        printf("  easily exceed 65504 -> NaN/Inf -> training divergence.\n");
        printf("  BF16 avoids this entirely without loss scaling.\n");
    }

    // ── B: Elementwise throughput ─────────────────────────────────────────────
    section("B: Elementwise kernel throughput by precision");
    {
        const int N  = 1 << 25;   // 32 M elements
        const int BLK = 256;
        const int GRID = (N + BLK - 1) / BLK;

        // FP32
        float *d_f32_in, *d_f32_out;
        CUDA_CHECK(cudaMalloc(&d_f32_in,  (size_t)N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_f32_out, (size_t)N * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_f32_in, 1, (size_t)N * sizeof(float)));

        float ms_fp32 = bench([&](){
            scaleKernelFP32<<<GRID, BLK>>>(d_f32_in, d_f32_out, 1.0001f, N);
        });

        // FP16
        __half *d_f16_in, *d_f16_out;
        CUDA_CHECK(cudaMalloc(&d_f16_in,  (size_t)N * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_f16_out, (size_t)N * sizeof(__half)));
        CUDA_CHECK(cudaMemset(d_f16_in, 1, (size_t)N * sizeof(__half)));

        float ms_fp16 = bench([&](){
            scaleKernelFP16<<<GRID, BLK>>>(d_f16_in, d_f16_out, __float2half(1.0001f), N);
        });

        printf("  %-8s %10s %12s %12s\n",
               "Format", "Time(ms)", "BW(GB/s)", "vs FP32");
        printf("  %-8s %10s %12s %12s\n",
               "--------","--------","----------","--------");

        double bw32 = bw_gb_s(2ull * N * sizeof(float), ms_fp32);
        double bw16 = bw_gb_s(2ull * N * sizeof(__half), ms_fp16);

        printf("  %-8s %10.3f %12.1f  1.00x\n", "FP32", ms_fp32, bw32);
        printf("  %-8s %10.3f %12.1f  %.2fx\n",  "FP16", ms_fp16, bw16, bw32/bw16);

        if (has_bf16) {
            __nv_bfloat16 *d_bf16_in, *d_bf16_out;
            CUDA_CHECK(cudaMalloc(&d_bf16_in,  (size_t)N * sizeof(__nv_bfloat16)));
            CUDA_CHECK(cudaMalloc(&d_bf16_out, (size_t)N * sizeof(__nv_bfloat16)));
            CUDA_CHECK(cudaMemset(d_bf16_in, 1, (size_t)N * sizeof(__nv_bfloat16)));

            // BF16 and FP16 are the same width (2 bytes/element), so their
            // memory bandwidth is identical. We reuse the FP16 kernel to measure
            // raw HBM throughput — the result is the same as a native BF16 kernel.
            float ms_bf16 = bench([&](){
                scaleKernelFP16<<<GRID, BLK>>>(
                    reinterpret_cast<const __half*>(d_bf16_in),
                    reinterpret_cast<__half*>(d_bf16_out),
                    __float2half(1.0001f), N);
            });
            double bw_bf16 = bw_gb_s(2ull * N * sizeof(__nv_bfloat16), ms_bf16);
            printf("  %-8s %10.3f %12.1f  %.2fx  (same bandwidth as FP16)\n",
                   "BF16", ms_bf16, bw_bf16, bw32/ms_bf16);

            cudaFree(d_bf16_in); cudaFree(d_bf16_out);
        }

        printf("\n  Memory-bound kernels scale with element size.\n");
        printf("  FP16/BF16 = 2 bytes/elem -> ~2x throughput over FP32 (4 bytes).\n");
        printf("  The real leverage comes from Tensor Cores (see program 02).\n");

        cudaFree(d_f32_in); cudaFree(d_f32_out);
        cudaFree(d_f16_in); cudaFree(d_f16_out);
    }

    // ── C: Matmul throughput: FP32 CUDA cores vs FP16 Tensor Cores ───────────
    section("C: Matmul throughput -- FP32 CUDA cores vs FP16 Tensor Cores");
    {
        const int M = 4096, N = 4096, K = 4096;
        double flops = 2.0 * M * N * K;

        cublasHandle_t handle;
        CUBLAS_CHECK(cublasCreate(&handle));

        // FP32 matmul (CUDA cores)
        float *d_A32, *d_B32, *d_C32;
        CUDA_CHECK(cudaMalloc(&d_A32, (size_t)M * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B32, (size_t)K * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C32, (size_t)M * N * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_A32, 1, (size_t)M * K * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_B32, 1, (size_t)K * N * sizeof(float)));

        float alpha32 = 1.f, beta32 = 0.f;
        float ms_fp32 = bench([&](){
            CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K, &alpha32, d_B32, N, d_A32, K, &beta32, d_C32, N));
        }, 20);

        // FP16 matmul (Tensor Cores) using cublasHgemm
        __half *d_A16, *d_B16, *d_C16;
        CUDA_CHECK(cudaMalloc(&d_A16, (size_t)M * K * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_B16, (size_t)K * N * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_C16, (size_t)M * N * sizeof(__half)));
        CUDA_CHECK(cudaMemset(d_A16, 1, (size_t)M * K * sizeof(__half)));
        CUDA_CHECK(cudaMemset(d_B16, 1, (size_t)K * N * sizeof(__half)));

        __half alpha16 = __float2half(1.f), beta16 = __float2half(0.f);
        float ms_fp16 = bench([&](){
            CUBLAS_CHECK(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K, &alpha16, d_B16, N, d_A16, K, &beta16, d_C16, N));
        }, 20);

        printf("  Matrix: M=%d N=%d K=%d  (%.1f GFLOP)\n\n", M, N, K, flops/1e9);
        printf("  %-28s %10s %12s %10s\n",
               "Method", "Time(ms)", "GFLOP/s", "vs FP32");
        printf("  %-28s %10s %12s %10s\n",
               "----------------------------","--------","----------","--------");

        double gf32 = gflops(flops, ms_fp32);
        double gf16 = gflops(flops, ms_fp16);

        printf("  %-28s %10.2f %12.1f  1.00x\n",
               "FP32 (CUDA cores)", ms_fp32, gf32);
        printf("  %-28s %10.2f %12.1f  %.1fx\n",
               "FP16 (Tensor Cores)", ms_fp16, gf16, gf32 / gf16);

        printf("\n  The Tensor Core speedup is the most important number in\n");
        printf("  GPU ML performance. Every LLM forward pass lives here.\n\n");
        printf("  Tensor Cores REQUIRE:\n");
        printf("    - FP16 or BF16 (not FP32) input types\n");
        printf("    - Matrix dimensions that are multiples of 16\n");
        printf("    - Row-major A, column-major B layout (cuBLAS handles this)\n");
        printf("  If any condition fails, cuBLAS silently falls back to CUDA\n");
        printf("  cores at 1/15th the speed with NO error or warning.\n");

        cublasDestroy(handle);
        cudaFree(d_A32); cudaFree(d_B32); cudaFree(d_C32);
        cudaFree(d_A16); cudaFree(d_B16); cudaFree(d_C16);
    }

    // ── D: Precision loss in optimizer state ──────────────────────────────────
    section("D: Why optimizer state must stay in FP32");
    if (has_bf16) {
        float* d_results;
        CUDA_CHECK(cudaMalloc(&d_results, 3 * sizeof(float)));
        precisionLossDemo<<<1, 1>>>(d_results);
        float results[3];
        CUDA_CHECK(cudaMemcpy(results, d_results, 3 * sizeof(float),
                              cudaMemcpyDeviceToHost));
        printf("  Accumulating 1000 updates of 0.001 onto base value 1000.0:\n\n");
        printf("  Expected result           : %.4f\n",  results[2]);
        printf("  FP32 accumulation         : %.4f  error=%.4e\n",
               results[0], fabsf(results[0] - results[2]));
        printf("  BF16 accumulation         : %.4f  error=%.4e\n",
               results[1], fabsf(results[1] - results[2]));
        printf("\n  BF16 has 7 mantissa bits -> ~1%% relative precision.\n");
        printf("  A gradient of 0.001 on a weight of 1000 is a 0.0001%% update.\n");
        printf("  BF16 cannot represent this difference: it rounds to zero.\n");
        printf("  After 1000 steps of dead updates, training plateaus early.\n\n");
        printf("  Solution: keep weight and optimizer state in FP32 ('master copy').\n");
        printf("  Cast to BF16 only for the forward/backward passes on Tensor Cores.\n");
        printf("  Memory cost: 6 bytes/param (FP32 master + BF16 working copy)\n");
        printf("  vs 4 bytes/param (FP32 only). Worth every byte.\n");
        cudaFree(d_results);
    } else {
        printf("  (BF16 requires sm_80+, skipping on this device)\n");
    }

    return 0;
}
