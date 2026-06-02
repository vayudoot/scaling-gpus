// 05_w4a16.cu  --  Post 7: Mixed Precision and Quantization
//
// W4A16 (4-bit weights, 16-bit activations) is the dominant quantization
// scheme for LLM inference because it targets the decode bottleneck:
//
//   During decode (batch size 1), arithmetic intensity ~ 1 FLOPs/byte.
//   The GPU is limited entirely by how fast it can load weight bytes from HBM.
//   INT4 weights use 0.5 bytes/param instead of 2 bytes (FP16):
//     4x fewer bytes loaded -> 4x more effective memory bandwidth -> 4x faster.
//
// This program:
//   A. Implements per-group INT4 quantization (the standard for accuracy).
//      Group size G=128: one scale per 128 consecutive weights.
//      This is what GPTQ, AWQ, llama.cpp, and bitsandbytes 4-bit use.
//
//   B. W4A16 dequant-then-matmul kernel: unpack INT4 weights, dequantize to
//      FP16 on the fly, multiply with FP16 activations, accumulate in FP32.
//
//   C. Bandwidth analysis: shows why 4x fewer bytes = ~4x faster decode.
//
//   D. Memory savings: 70B model at FP16 = 140 GB. At INT4 = 35 GB.
//      Fits on one H100 instead of two.
//
// INT4 packing: two 4-bit values stored per byte.
//   byte = (hi_nibble << 4) | lo_nibble
//   value = (byte >> (nibble_idx * 4)) & 0xF
//   Signed INT4 range: -8 to 7. We use offset coding: store val+8 (0-15)
//   and subtract 8 during dequant.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include "../include/utils.cuh"

#define GROUP_SIZE 128   // standard per-group quantization group size

// ─────────────────────────────────────────────────────────────────────────────
// INT4 packing helpers (CPU)
// ─────────────────────────────────────────────────────────────────────────────

// Pack two int4 values into one byte. val must be in [-8, 7] (signed INT4).
// We store as unsigned with offset 8: stored = val + 8  (range 0-15).
static inline uint8_t pack_int4(int lo, int hi) {
    uint8_t lo_u = (uint8_t)((lo + 8) & 0xF);
    uint8_t hi_u = (uint8_t)((hi + 8) & 0xF);
    return lo_u | (hi_u << 4);
}

static inline __host__ __device__ int unpack_int4_lo(uint8_t packed) { return (int)(packed & 0xF) - 8; }
static inline __host__ __device__ int unpack_int4_hi(uint8_t packed) { return (int)(packed >> 4)  - 8; }

// ─────────────────────────────────────────────────────────────────────────────
// Per-group INT4 quantization (CPU, runs once at model load time)
// ─────────────────────────────────────────────────────────────────────────────
// Weight matrix W [N_out x K_in] is quantized with:
//   - One scale per group of G consecutive weights along the K dimension
//   - scales: [N_out x (K_in / G)]  float16
//   - packed: [N_out x (K_in / 2)]  uint8  (two INT4 per byte)
static void quantizePerGroupInt4(
    const float* W,         // [N_out x K_in] fp32 weights
    uint8_t*     packed,    // [N_out x K_in/2] output packed INT4
    float*       scales_fp32,// [N_out x K_in/G] output scales (fp32)
    int N_out, int K_in, int G)
{
    int num_groups = K_in / G;
    for (int n = 0; n < N_out; n++) {
        for (int g = 0; g < num_groups; g++) {
            // Find absmax for this group
            float absmax = 0.f;
            for (int i = 0; i < G; i++)
                absmax = fmaxf(absmax, fabsf(W[n * K_in + g * G + i]));

            float scale = (absmax > 0.f) ? absmax / 7.f : 1.f;  // INT4 max = 7
            scales_fp32[n * num_groups + g] = scale;

            // Quantize and pack two values per byte
            for (int i = 0; i < G; i += 2) {
                float v0 = W[n * K_in + g * G + i];
                float v1 = W[n * K_in + g * G + i + 1];
                int q0 = (int)fminf(7.f, fmaxf(-8.f, rintf(v0 / scale)));
                int q1 = (int)fminf(7.f, fmaxf(-8.f, rintf(v1 / scale)));
                packed[n * (K_in/2) + (g * G + i) / 2] = pack_int4(q0, q1);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// W4A16 dequantize-then-matmul kernel
// ─────────────────────────────────────────────────────────────────────────────
// Each thread computes one output element: out[m][n] = sum_k( x[m][k] * w[n][k] )
// Weights are unpacked from INT4 and dequantized to FP16 on the fly.
// Accumulation is in FP32 for accuracy.
__global__ void w4a16MatmulKernel(
    const __half*   __restrict__ x,       // [M x K] FP16 activations
    const uint8_t*  __restrict__ w_int4,  // [N x K/2] packed INT4 weights
    const __half*   __restrict__ scales,  // [N x K/G] FP16 per-group scales
    __half*         __restrict__ out,     // [M x N] FP16 output
    int M, int N, int K, int G)
{
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || n >= N) return;

    int num_groups = K / G;
    float acc = 0.f;

    for (int g = 0; g < num_groups; g++) {
        float scale = __half2float(scales[n * num_groups + g]);
        int k_base  = g * G;

        for (int i = 0; i < G; i += 2) {
            int    k        = k_base + i;
            uint8_t packed   = w_int4[n * (K/2) + k/2];

            // Unpack and dequantize both elements of this byte
            float w0 = (float)unpack_int4_lo(packed) * scale;
            float w1 = (float)unpack_int4_hi(packed) * scale;

            // Multiply with FP16 activations, accumulate in FP32
            acc += __half2float(x[m * K + k])     * w0;
            acc += __half2float(x[m * K + k + 1]) * w1;
        }
    }
    out[m * N + n] = __float2half(acc);
}

// ─────────────────────────────────────────────────────────────────────────────
// FP16 reference matmul (for accuracy comparison)
// ─────────────────────────────────────────────────────────────────────────────
static void cpuMatmulFP16ref(const float* x_fp32, const float* W_fp32,
                               float* out, int M, int N, int K) {
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float acc = 0.f;
            for (int k = 0; k < K; k++) acc += x_fp32[m*K+k] * W_fp32[n*K+k];
            out[m*N+n] = acc;
        }
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    print_device_info();
    srand(42);

    // ── A: Bandwidth and memory savings analysis ──────────────────────────────
    section("A: Why W4A16 is optimal for LLM decode");
    {
        printf("  Decode (batch=1) arithmetic intensity:\n");
        printf("    AI = FLOPs / bytes = (2*N*K) / (N*K*bytes_per_weight) = 2/bytes\n\n");
        printf("  Format   bytes/w  AI(FLOPs/B)  HBM-limited to\n");
        printf("  ------   -------  -----------  --------------\n");
        printf("  FP32       4.0      0.5          3350/4 * 2 = 1675 GFLOP/s  (H100)\n");
        printf("  FP16       2.0      1.0          3350/2 * 2 = 3350 GFLOP/s\n");
        printf("  INT8       1.0      2.0          3350/1 * 2 = 6700 GOPS\n");
        printf("  INT4       0.5      4.0          3350/.5 * 2 = 13400 GOPS\n\n");
        printf("  At batch=1, the GPU loads all weights once per decode step.\n");
        printf("  INT4: load 4x fewer bytes -> 4x more tokens/second.\n\n");

        // Memory footprint of common LLMs
        printf("  LLM memory footprint (weights only):\n");
        printf("  %-12s  %-8s  %-8s  %-8s  %-8s\n",
               "Model", "FP32", "FP16", "INT8", "INT4");
        printf("  %-12s  %-8s  %-8s  %-8s  %-8s\n",
               "------------","--------","--------","--------","--------");
        struct { const char* name; double params; } models[] = {
            {"7B",   7e9}, {"13B",  13e9}, {"70B",  70e9}, {"405B", 405e9}
        };
        for (auto& m : models) {
            printf("  %-12s  %5.0f GB  %5.0f GB  %5.0f GB  %5.0f GB\n",
                   m.name,
                   m.params * 4 / 1e9,
                   m.params * 2 / 1e9,
                   m.params * 1 / 1e9,
                   m.params * 0.5 / 1e9);
        }
        printf("\n  70B model: FP16 needs 2xH100 (140 GB); INT4 fits on 1xH100 (35 GB).\n");
    }

    // ── B: Correctness check ──────────────────────────────────────────────────
    section("B: Per-group INT4 quantization accuracy");
    {
        const int N_OUT = 64, K_IN = 512;
        const int M     = 4;   // batch size

        float* h_W     = (float*)malloc((size_t)N_OUT * K_IN * sizeof(float));
        float* h_x     = (float*)malloc((size_t)M     * K_IN * sizeof(float));
        rand_fill_fp32(h_W, N_OUT * K_IN, -0.5f, 0.5f);
        rand_fill_fp32(h_x, M     * K_IN, -0.5f, 0.5f);

        // CPU FP32 reference
        float* h_out_ref = (float*)malloc((size_t)M * N_OUT * sizeof(float));
        cpuMatmulFP16ref(h_x, h_W, h_out_ref, M, N_OUT, K_IN);

        // Quantize weights to INT4 per-group
        int num_groups = K_IN / GROUP_SIZE;
        uint8_t* h_packed    = (uint8_t*)malloc((size_t)N_OUT * K_IN / 2);
        float*   h_scales_fp = (float*)  malloc((size_t)N_OUT * num_groups * sizeof(float));
        quantizePerGroupInt4(h_W, h_packed, h_scales_fp, N_OUT, K_IN, GROUP_SIZE);

        // Convert scales to FP16
        __half* h_scales_fp16 = (__half*)malloc((size_t)N_OUT * num_groups * sizeof(__half));
        for (int i = 0; i < N_OUT * num_groups; i++)
            h_scales_fp16[i] = __float2half(h_scales_fp[i]);

        // Convert activations to FP16
        __half* h_x16 = (__half*)malloc((size_t)M * K_IN * sizeof(__half));
        fp32_to_fp16(h_x, h_x16, M * K_IN);

        // Copy to device
        uint8_t* d_packed;
        __half   *d_x16, *d_scales, *d_out;
        CUDA_CHECK(cudaMalloc(&d_packed, (size_t)N_OUT * K_IN / 2));
        CUDA_CHECK(cudaMalloc(&d_x16,    (size_t)M * K_IN * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_scales, (size_t)N_OUT * num_groups * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_out,    (size_t)M * N_OUT * sizeof(__half)));

        CUDA_CHECK(cudaMemcpy(d_packed, h_packed,    (size_t)N_OUT * K_IN / 2,                      cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_x16,   h_x16,       (size_t)M * K_IN * sizeof(__half),              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_scales,h_scales_fp16,(size_t)N_OUT * num_groups * sizeof(__half),   cudaMemcpyHostToDevice));

        dim3 blk(16, 4);
        dim3 grd((N_OUT+15)/16, (M+3)/4);
        w4a16MatmulKernel<<<grd, blk>>>(d_x16, d_packed, d_scales, d_out, M, N_OUT, K_IN, GROUP_SIZE);
        CUDA_CHECK(cudaDeviceSynchronize());

        __half* h_out_fp16 = (__half*)malloc((size_t)M * N_OUT * sizeof(__half));
        CUDA_CHECK(cudaMemcpy(h_out_fp16, d_out, (size_t)M * N_OUT * sizeof(__half), cudaMemcpyDeviceToHost));
        float* h_out_fp32 = (float*)malloc((size_t)M * N_OUT * sizeof(float));
        fp16_to_fp32(h_out_fp16, h_out_fp32, M * N_OUT);

        // Compute error
        float max_err = 0.f, sum_sq = 0.f;
        for (int i = 0; i < M * N_OUT; i++) {
            float e = fabsf(h_out_fp32[i] - h_out_ref[i]);
            max_err = fmaxf(max_err, e);
            sum_sq += e * e;
        }
        float rmse = sqrtf(sum_sq / (M * N_OUT));
        float ref_scale = 0.f;
        for (int i = 0; i < M * N_OUT; i++) ref_scale = fmaxf(ref_scale, fabsf(h_out_ref[i]));
        float rel_err = (ref_scale > 0) ? max_err / ref_scale : 0.f;

        printf("  N_out=%d K_in=%d M=%d  group_size=%d\n", N_OUT, K_IN, M, GROUP_SIZE);
        printf("  Max absolute error : %.4e\n", (double)max_err);
        printf("  RMSE               : %.4e\n", (double)rmse);
        printf("  Relative max error : %.2f%%\n", (double)(rel_err * 100.f));
        printf("  %s\n\n",
               rel_err < 0.05f ? "PASS -- within 5% relative error" :
               "FAIL -- error too large, check packing");
        printf("  Per-group G=%d means one scale per 128 weights.\n", GROUP_SIZE);
        printf("  This is the key to INT4 accuracy: per-group captures\n");
        printf("  local weight distribution better than per-tensor or per-channel.\n");

        cudaFree(d_packed); cudaFree(d_x16); cudaFree(d_scales); cudaFree(d_out);
        free(h_W); free(h_x); free(h_out_ref); free(h_packed);
        free(h_scales_fp); free(h_scales_fp16); free(h_x16);
        free(h_out_fp16); free(h_out_fp32);
    }

    // ── C: Bandwidth-limited decode throughput ────────────────────────────────
    section("C: Decode throughput: FP16 vs W4A16 weight loading");
    {
        // Simulate a single decode step: load a weight matrix from HBM.
        // At batch=1, time is dominated by the memory load, not arithmetic.
        const int N_OUT = 4096, K_IN = 4096;
        const int M     = 1;   // batch=1 (decode)

        size_t bytes_fp16 = (size_t)N_OUT * K_IN * sizeof(__half);
        size_t bytes_int4 = (size_t)N_OUT * K_IN / 2;   // 0.5 bytes per weight

        // Allocate and warm up
        __half   *d_W_fp16, *d_x16, *d_out_fp16;
        uint8_t  *d_W_int4;
        __half   *d_scales;
        int num_groups = K_IN / GROUP_SIZE;

        CUDA_CHECK(cudaMalloc(&d_W_fp16,   bytes_fp16));
        CUDA_CHECK(cudaMalloc(&d_x16,      (size_t)M * K_IN * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_out_fp16, (size_t)M * N_OUT * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_W_int4,   bytes_int4));
        CUDA_CHECK(cudaMalloc(&d_scales,   (size_t)N_OUT * num_groups * sizeof(__half)));
        CUDA_CHECK(cudaMemset(d_W_fp16, 1, bytes_fp16));
        CUDA_CHECK(cudaMemset(d_x16, 1, (size_t)M * K_IN * sizeof(__half)));
        CUDA_CHECK(cudaMemset(d_W_int4, 1, bytes_int4));
        CUDA_CHECK(cudaMemset(d_scales, 1, (size_t)N_OUT * num_groups * sizeof(__half)));

        cublasHandle_t cublas;
        CUBLAS_CHECK(cublasCreate(&cublas));
        __half alpha16 = __float2half(1.f), beta16 = __float2half(0.f);

        const int REPS = 100;
        GpuTimer t;

        // FP16 matmul (standard cublasHgemm)
        cublasHgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                    N_OUT, M, K_IN, &alpha16, d_W_fp16, K_IN,
                    d_x16, K_IN, &beta16, d_out_fp16, N_OUT);
        CUDA_CHECK(cudaDeviceSynchronize());

        t.start();
        for (int r = 0; r < REPS; r++)
            cublasHgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                        N_OUT, M, K_IN, &alpha16, d_W_fp16, K_IN,
                        d_x16, K_IN, &beta16, d_out_fp16, N_OUT);
        float ms_fp16 = t.stop_ms() / REPS;

        // W4A16 kernel
        dim3 blk(16, 1);
        dim3 grd((N_OUT+15)/16, M);
        w4a16MatmulKernel<<<grd, blk>>>(d_x16, d_W_int4, d_scales, d_out_fp16,
                                         M, N_OUT, K_IN, GROUP_SIZE);
        CUDA_CHECK(cudaDeviceSynchronize());

        t.start();
        for (int r = 0; r < REPS; r++)
            w4a16MatmulKernel<<<grd, blk>>>(d_x16, d_W_int4, d_scales, d_out_fp16,
                                             M, N_OUT, K_IN, GROUP_SIZE);
        float ms_w4a16 = t.stop_ms() / REPS;

        printf("  Linear layer: [%d x %d] x [%d x %d] = [%d x %d]  batch=1\n\n",
               M, K_IN, K_IN, N_OUT, M, N_OUT);
        printf("  %-22s %10s %12s %10s %12s\n",
               "Method", "Time(ms)", "Weight BW", "vs FP16", "w bytes");
        printf("  %-22s %10s %12s %10s %12s\n",
               "----------------------","--------","----------","--------","--------");

        double bw_fp16  = bw_gb_s(bytes_fp16, ms_fp16);
        double bw_w4a16 = bw_gb_s(bytes_int4, ms_w4a16);  // effective BW on weight bytes

        printf("  %-22s %10.3f %10.1f GB/s  1.00x  %8zu MB\n",
               "FP16 (Tensor Cores)", ms_fp16, bw_fp16, bytes_fp16 / (1024*1024));
        printf("  %-22s %10.3f %10.1f GB/s  %.2fx  %8zu MB\n",
               "W4A16 (dequant+mul)", ms_w4a16, bw_w4a16,
               ms_fp16 / ms_w4a16, bytes_int4 / (1024*1024));

        printf("\n  Note: our W4A16 kernel is pedagogical, not production-optimized.\n");
        printf("  Production kernels (GPTQ, AWQ, llama.cpp) use:\n");
        printf("    - Tiled shared memory loading\n");
        printf("    - Vectorized uint4 unpacking (8 values per ldg.128)\n");
        printf("    - FP16 or BF16 Tensor Core accumulation per tile\n");
        printf("    - On H100: ~3-4x FP16 throughput at batch=1\n");

        cublasDestroy(cublas);
        cudaFree(d_W_fp16); cudaFree(d_x16); cudaFree(d_out_fp16);
        cudaFree(d_W_int4); cudaFree(d_scales);
    }

    section("W4A16 vs alternatives: decision matrix");
    printf("  Scenario                        Recommendation\n");
    printf("  -------------------------------- ---------------\n");
    printf("  Batch=1 decode, any model size   W4A16 (max bandwidth)\n");
    printf("  Batch>32 decode                  W8A8 or FP16 (compute-bound)\n");
    printf("  Model fits in GPU VRAM at FP16   FP16 (no accuracy loss)\n");
    printf("  Model too large at FP16          W4A16 (4x compression)\n");
    printf("  Maximum accuracy required        FP16 or BF16\n");
    printf("  H100 with FP8 support            FP8 E4M3 (best throughput)\n");

    return 0;
}
