// 04_quantization.cu  --  Post 7: Mixed Precision and Quantization
//
// Quantization maps floating-point values to integers:
//   - Halves (INT8) or quarters (INT4) the bytes per weight
//   - Enables hardware integer instructions (dp4a) that are faster than FP16
//   - Requires a "scale factor" to recover approximate FP32 values
//
// This program implements and benchmarks:
//   A. Symmetric INT8 quantization:
//      scale = max(|x|) / 127
//      x_int8 = round(clamp(x / scale, -127, 127))
//      x_dequant = x_int8 * scale
//
//   B. Per-tensor vs per-channel quantization:
//      Per-tensor:  one scale for the entire weight matrix
//      Per-channel: one scale per output channel (row)
//      Per-channel is more accurate because outliers in one channel
//      don't corrupt the scale for all other channels.
//
//   C. dp4a: CUDA's hardware INT8 dot product instruction.
//      Computes a += b0*c0 + b1*c1 + b2*c2 + b3*c3 in one instruction.
//      Four INT8 values packed into one INT32 register.
//      This is the building block for INT8 matrix multiplication.
//
//   D. INT8 matmul using dp4a + dequantization.
//
// Accuracy analysis: shows the quantization error at different granularities.

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include "../include/utils.cuh"

// ─────────────────────────────────────────────────────────────────────────────
// Quantization kernels
// ─────────────────────────────────────────────────────────────────────────────

// Per-tensor symmetric quantization:
//   Assumes absmax has been computed and stored in d_scale[0] by the caller.
//   After this kernel, d_scale[0] holds the scale factor (absmax / 127).
__global__ void quantizeSymmetricPerTensor(
    const float*   __restrict__ fp32,
    int8_t*        __restrict__ int8,
    const float*   __restrict__ d_absmax,   // [1] -- max(|fp32|)
    float*         __restrict__ d_scale,    // [1] -- output: absmax/127
    int n)
{
    float absmax = d_absmax[0];
    float scale  = absmax / 127.f;
    if (threadIdx.x == 0 && blockIdx.x == 0) d_scale[0] = scale;
    __syncthreads();

    float inv_scale = (scale > 0.f) ? 127.f / absmax : 0.f;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = fp32[i] * inv_scale;
        v = fminf(127.f, fmaxf(-127.f, rintf(v)));
        int8[i] = (int8_t)(int)v;
    }
}

// Per-channel quantization: each output channel (row) gets its own scale.
// d_scale[row] = max(|fp32[row,:]|) / 127
__global__ void quantizeSymmetricPerChannel(
    const float*   __restrict__ fp32,   // [rows x cols]
    int8_t*        __restrict__ int8,
    float*         __restrict__ d_scale, // [rows]
    int rows, int cols)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    extern __shared__ float smem[];

    // Find absmax for this row
    float local_max = 0.f;
    for (int c = tid; c < cols; c += blockDim.x)
        local_max = fmaxf(local_max, fabsf(fp32[row * cols + c]));
    smem[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid+s]);
        __syncthreads();
    }
    float absmax = smem[0];
    float scale  = absmax / 127.f;
    if (tid == 0) d_scale[row] = scale;
    __syncthreads();

    float inv_scale = (absmax > 0.f) ? 127.f / absmax : 0.f;
    for (int c = tid; c < cols; c += blockDim.x) {
        float v = fp32[row * cols + c] * inv_scale;
        v = fminf(127.f, fmaxf(-127.f, rintf(v)));
        int8[row * cols + c] = (int8_t)(int)v;
    }
}

// Dequantize INT8 back to FP32 (per-tensor)
__global__ void dequantizePerTensor(
    const int8_t* __restrict__ int8,
    float*        __restrict__ fp32,
    float scale, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) fp32[i] = (float)int8[i] * scale;
}

// Reduce to find absmax (parallel reduction, single block for small arrays)
__global__ void findAbsmax(const float* __restrict__ data,
                             float*       __restrict__ absmax,
                             int n) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    float local = 0.f;
    for (int i = tid; i < n; i += blockDim.x)
        local = fmaxf(local, fabsf(data[i]));
    smem[tid] = local;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid+s]);
        __syncthreads();
    }
    if (tid == 0) absmax[0] = smem[0];
}

// ─────────────────────────────────────────────────────────────────────────────
// INT8 matmul using dp4a
// ─────────────────────────────────────────────────────────────────────────────
// __dp4a: dot product of 4 INT8 pairs, accumulated into INT32.
//   int __dp4a(int srcA, int srcB, int c) --> c + dot4(srcA, srcB)
// srcA and srcB are 4 INT8 values packed into INT32 (little-endian).
//
// This kernel:
//   - A [M x K] INT8 row-major
//   - B [K x N] INT8 col-major (each column contiguous)
//   - C [M x N] INT32 output (needs dequantization after)
//
// Each thread computes one C[m,n] by iterating K/4 dp4a operations.

__global__ void int8MatmulDP4A(const int8_t* __restrict__ A,   // [M x K] row-major
                                 const int8_t* __restrict__ B,   // [K x N] col-major
                                 int32_t*      __restrict__ C,   // [M x N]
                                 int M, int N, int K) {
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || n >= N) return;

    int32_t acc = 0;
    // Process K in groups of 4 (dp4a operates on 4 elements at once)
    for (int k = 0; k < K; k += 4) {
        // Pack 4 INT8 values from row m of A into one INT32
        int a_pack = *reinterpret_cast<const int*>(&A[m * K + k]);
        // Pack 4 INT8 values from column n of B (col-major: stride = 1)
        int b_pack = *reinterpret_cast<const int*>(&B[n * K + k]);
        acc = __dp4a(a_pack, b_pack, acc);
    }
    C[m * N + n] = acc;
}

// Apply per-tensor dequantization to INT32 matmul output
__global__ void dequantizeMatmulOutput(const int32_t* __restrict__ int32,
                                        float*         __restrict__ fp32,
                                        float scale_a, float scale_b,
                                        int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) fp32[i] = (float)int32[i] * scale_a * scale_b;
}

// ─────────────────────────────────────────────────────────────────────────────
// CPU reference
// ─────────────────────────────────────────────────────────────────────────────
static void cpuMatmulFP32(const float* A, const float* B, float* C,
                            int M, int N, int K) {
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float s = 0.f;
            for (int k = 0; k < K; k++) s += A[m*K+k] * B[k*N+n];
            C[m*N+n] = s;
        }
}

// ─────────────────────────────────────────────────────────────────────────────
// Quantization error analysis
// ─────────────────────────────────────────────────────────────────────────────
static void printQuantizationError(const char* label,
                                    const float* original, const float* dequant,
                                    int n) {
    float max_err = 0.f, sum_sq = 0.f;
    for (int i = 0; i < n; i++) {
        float e = fabsf(original[i] - dequant[i]);
        max_err  = fmaxf(max_err, e);
        sum_sq  += e * e;
    }
    float rmse = sqrtf(sum_sq / n);
    printf("  %-30s  max_err=%.4e  RMSE=%.4e\n", label, (double)max_err, (double)rmse);
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main() {
    print_device_info();
    srand(42);

    // ── A: Quantization accuracy comparison ──────────────────────────────────
    section("A: Per-tensor vs per-channel quantization accuracy");
    {
        const int ROWS = 128, COLS = 256;
        const int N    = ROWS * COLS;
        const int BLK  = 256;

        float*   h_fp32    = (float*)  malloc((size_t)N * sizeof(float));
        int8_t*  h_int8    = (int8_t*) malloc((size_t)N * sizeof(int8_t));
        float*   h_dequant = (float*)  malloc((size_t)N * sizeof(float));

        // Generate data with some large outliers (realistic neural network weights)
        rand_fill_fp32(h_fp32, N, -0.5f, 0.5f);
        // Add a few large outliers -- common in LLM weights
        for (int i = 0; i < 5; i++) {
            int row = rand() % ROWS;
            h_fp32[row * COLS + rand() % COLS] = (float)(rand() % 2 == 0 ? 10.f : -10.f);
        }

        float *d_fp32, *d_dequant, *d_scale, *d_absmax;
        float *d_scale_ch;
        int8_t *d_int8_tensor, *d_int8_channel;

        CUDA_CHECK(cudaMalloc(&d_fp32,         (size_t)N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dequant,       (size_t)N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_scale,         sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_absmax,        sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_scale_ch,      (size_t)ROWS * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_int8_tensor,   (size_t)N * sizeof(int8_t)));
        CUDA_CHECK(cudaMalloc(&d_int8_channel,  (size_t)N * sizeof(int8_t)));

        CUDA_CHECK(cudaMemcpy(d_fp32, h_fp32, (size_t)N * sizeof(float), cudaMemcpyHostToDevice));

        // Per-tensor quantization
        findAbsmax<<<1, BLK, BLK * sizeof(float)>>>(d_fp32, d_absmax, N);
        quantizeSymmetricPerTensor<<<(N+BLK-1)/BLK, BLK>>>(
            d_fp32, d_int8_tensor, d_absmax, d_scale, N);
        float h_scale;
        CUDA_CHECK(cudaMemcpy(&h_scale, d_scale, sizeof(float), cudaMemcpyDeviceToHost));
        dequantizePerTensor<<<(N+BLK-1)/BLK, BLK>>>(d_int8_tensor, d_dequant, h_scale, N);
        CUDA_CHECK(cudaMemcpy(h_dequant, d_dequant, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost));
        printQuantizationError("Per-tensor INT8", h_fp32, h_dequant, N);

        // Per-channel quantization
        quantizeSymmetricPerChannel<<<ROWS, BLK, BLK*sizeof(float)>>>(
            d_fp32, d_int8_channel, d_scale_ch, ROWS, COLS);
        // Dequantize using per-channel scales
        {
            float* h_scales_ch = (float*)malloc(ROWS * sizeof(float));
            CUDA_CHECK(cudaMemcpy(h_scales_ch, d_scale_ch, ROWS*sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_int8, d_int8_channel, (size_t)N*sizeof(int8_t), cudaMemcpyDeviceToHost));
            for (int r = 0; r < ROWS; r++)
                for (int c = 0; c < COLS; c++)
                    h_dequant[r*COLS+c] = (float)h_int8[r*COLS+c] * h_scales_ch[r];
            free(h_scales_ch);
        }
        printQuantizationError("Per-channel INT8", h_fp32, h_dequant, N);

        printf("\n  Per-channel is more accurate because outliers in one channel\n");
        printf("  don't force a conservative scale that wastes precision elsewhere.\n");
        printf("  The scale overhead: per-tensor=1 float, per-channel=%d floats.\n", ROWS);
        printf("  For a 4096x4096 weight: per-channel adds only 4096*4B=16 KB.\n");

        cudaFree(d_fp32); cudaFree(d_dequant); cudaFree(d_scale); cudaFree(d_absmax);
        cudaFree(d_scale_ch); cudaFree(d_int8_tensor); cudaFree(d_int8_channel);
        free(h_fp32); free(h_int8); free(h_dequant);
    }

    // ── B: dp4a INT8 matmul benchmark ─────────────────────────────────────────
    section("B: INT8 matmul with dp4a vs FP32 matmul");
    {
        const int M = 1024, N = 1024, K = 1024;
        const int BLK = 16;  // threads per dim for the matmul kernel

        // Must be a multiple of 4 for dp4a
        static_assert(K % 4 == 0, "K must be multiple of 4 for dp4a");

        float*   h_A    = (float*)  malloc((size_t)M * K * sizeof(float));
        float*   h_B    = (float*)  malloc((size_t)K * N * sizeof(float));
        float*   h_C_fp32 = (float*)malloc((size_t)M * N * sizeof(float));
        float*   h_C_int8 = (float*)malloc((size_t)M * N * sizeof(float));

        rand_fill_fp32(h_A, M*K, -1.f, 1.f);
        rand_fill_fp32(h_B, K*N, -1.f, 1.f);
        // Small CPU reference for correctness (only a 64x64 submatrix)
        const int REF = 64;
        float* h_C_ref = (float*)malloc(REF * REF * sizeof(float));
        cpuMatmulFP32(h_A, h_B, h_C_ref, REF, REF, K);

        // Quantize A and B to INT8 (per-tensor)
        int8_t* h_A_int8 = (int8_t*)malloc((size_t)M * K * sizeof(int8_t));
        int8_t* h_B_int8_col = (int8_t*)malloc((size_t)K * N * sizeof(int8_t));

        // Find absmax and quantize on CPU for simplicity
        float absmax_a = 0.f, absmax_b = 0.f;
        for (int i = 0; i < M*K; i++) absmax_a = fmaxf(absmax_a, fabsf(h_A[i]));
        for (int i = 0; i < K*N; i++) absmax_b = fmaxf(absmax_b, fabsf(h_B[i]));
        float scale_a = absmax_a / 127.f;
        float scale_b = absmax_b / 127.f;
        for (int i = 0; i < M*K; i++) {
            float v = h_A[i] / scale_a;
            h_A_int8[i] = (int8_t)(int)fminf(127.f, fmaxf(-127.f, rintf(v)));
        }
        // B stored col-major for dp4a: B_col[n][k] = B_row[k][n]
        for (int k = 0; k < K; k++)
            for (int n = 0; n < N; n++) {
                float v = h_B[k*N+n] / scale_b;
                h_B_int8_col[n*K+k] = (int8_t)(int)fminf(127.f, fmaxf(-127.f, rintf(v)));
            }

        // Device allocations
        float   *d_A_fp32, *d_B_fp32, *d_C_fp32;
        int8_t  *d_A_int8, *d_B_int8_col;
        int32_t *d_C_int32;
        float   *d_C_dequant;

        CUDA_CHECK(cudaMalloc(&d_A_fp32,     (size_t)M*K*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B_fp32,     (size_t)K*N*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C_fp32,     (size_t)M*N*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_A_int8,     (size_t)M*K*sizeof(int8_t)));
        CUDA_CHECK(cudaMalloc(&d_B_int8_col, (size_t)K*N*sizeof(int8_t)));
        CUDA_CHECK(cudaMalloc(&d_C_int32,    (size_t)M*N*sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_C_dequant,  (size_t)M*N*sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_A_fp32,     h_A,          (size_t)M*K*sizeof(float),  cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B_fp32,     h_B,          (size_t)K*N*sizeof(float),  cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_A_int8,     h_A_int8,     (size_t)M*K*sizeof(int8_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B_int8_col, h_B_int8_col, (size_t)K*N*sizeof(int8_t), cudaMemcpyHostToDevice));

        cublasHandle_t cublas;
        CUBLAS_CHECK(cublasCreate(&cublas));
        float alpha = 1.f, beta = 0.f;

        // Warm up
        // Warm up both kernels before timing
        cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                    &alpha, d_B_fp32, N, d_A_fp32, K, &beta, d_C_fp32, N);
        dim3 blk(BLK, BLK), grd((N+BLK-1)/BLK, (M+BLK-1)/BLK);
        int8MatmulDP4A<<<grd, blk>>>(d_A_int8, d_B_int8_col, d_C_int32, M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());

        const int REPS = 50;
        GpuTimer t;

        t.start();
        for (int r = 0; r < REPS; r++)
            cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                        &alpha, d_B_fp32, N, d_A_fp32, K, &beta, d_C_fp32, N);
        float ms_fp32 = t.stop_ms() / REPS;

        t.start();
        for (int r = 0; r < REPS; r++) {
            int8MatmulDP4A<<<grd, blk>>>(d_A_int8, d_B_int8_col, d_C_int32, M, N, K);
            dequantizeMatmulOutput<<<(M*N+255)/256, 256>>>(
                d_C_int32, d_C_dequant, scale_a, scale_b, M*N);
        }
        float ms_int8 = t.stop_ms() / REPS;

        double flops = 2.0 * M * N * K;
        printf("  Matrix %dx%dx%d  (%.1f GFLOP)\n\n", M, N, K, flops/1e9);
        printf("  %-28s %10s %12s %10s\n",
               "Method", "Time(ms)", "GFLOP/s", "vs FP32");
        printf("  %-28s %10s %12s %10s\n",
               "----------------------------","--------","----------","--------");
        printf("  %-28s %10.3f %12.1f  1.00x\n",
               "FP32 cuBLAS",    ms_fp32, gflops(flops, ms_fp32));
        printf("  %-28s %10.3f %12.1f  %.2fx\n",
               "INT8 dp4a + dequant", ms_int8, gflops(flops, ms_int8),
               ms_fp32 / ms_int8);

        // Correctness
        CUDA_CHECK(cudaMemcpy(h_C_int8, d_C_dequant, (size_t)M*N*sizeof(float), cudaMemcpyDeviceToHost));
        printQuantizationError("\n  INT8 matmul vs FP32 reference", h_C_ref, h_C_int8, REF*REF);

        printf("\n  INT8 throughput advantage comes from:\n");
        printf("    - dp4a: 4 multiply-adds per instruction (vs 1 for FP32 FMA)\n");
        printf("    - Half the memory bandwidth: 1 byte/elem vs 4 bytes/elem\n");
        printf("    - INT32 accumulator avoids FP rounding in inner loop\n");

        cublasDestroy(cublas);
        cudaFree(d_A_fp32); cudaFree(d_B_fp32); cudaFree(d_C_fp32);
        cudaFree(d_A_int8); cudaFree(d_B_int8_col);
        cudaFree(d_C_int32); cudaFree(d_C_dequant);
        free(h_A); free(h_B); free(h_C_fp32); free(h_C_int8); free(h_C_ref);
        free(h_A_int8); free(h_B_int8_col);
    }

    section("Quantization decision guide");
    printf("  Format  Bytes/w  Accuracy  Use case\n");
    printf("  ------  -------  --------  --------\n");
    printf("  FP32      4.0    Exact     Training, reference\n");
    printf("  FP16      2.0    ~0.01%%    Inference, fast training\n");
    printf("  BF16      2.0    ~0.1%%     Training default (Ampere+)\n");
    printf("  INT8      1.0    ~0.5%%     Inference W8A8 (with calibration)\n");
    printf("  INT4      0.5    ~1-2%%     Weight-only quant W4A16 (LLM decode)\n\n");
    printf("  Rule: use the lowest precision that doesn't measurably hurt\n");
    printf("  your benchmark metric. INT4 with per-group scaling typically\n");
    printf("  costs <1%% on most LLM benchmarks.\n");
    return 0;
}
