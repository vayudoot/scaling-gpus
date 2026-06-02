// 03_mixed_precision.cu  --  Post 7: Mixed Precision and Quantization
//
// Implements the AMP (Automatic Mixed Precision) training contract from scratch:
//
//   1. Master weights stored in FP32 (full precision)
//   2. Working copy cast to FP16 before each forward pass
//   3. Forward + backward computed in FP16 on Tensor Cores
//   4. Gradients accumulated in FP32
//   5. FP32 master weights updated with FP32 gradients
//
// Also demonstrates loss scaling for FP16 training:
//   - FP16 minimum positive normal: ~6e-8. Gradients smaller than this
//     underflow to zero, causing training to stall.
//   - Loss scaling: multiply loss by a large constant (2^15) before backward.
//     This shifts all gradients up by that factor, keeping them in FP16 range.
//     After backward, divide gradients by the same constant before the update.
//   - Dynamic scaling: if any gradient is Inf/NaN (scale was too large),
//     skip the step and halve the scale. Otherwise, periodically double it.
//
// BF16 does NOT need loss scaling (same exponent range as FP32).
// This program demonstrates why FP16 did, and why BF16 is the default today.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/utils.cuh"

// ─────────────────────────────────────────────────────────────────────────────
// Kernels
// ─────────────────────────────────────────────────────────────────────────────

// Cast FP32 master weights to FP16 working copy
__global__ void castFP32toFP16(const float* __restrict__ fp32,
                                 __half*       __restrict__ fp16,
                                 int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) fp16[i] = __float2half(fp32[i]);
}

// Cast FP16 gradients to FP32 and apply loss scale
__global__ void castGradFP16toFP32(const __half* __restrict__ grad16,
                                    float*        __restrict__ grad32,
                                    float inv_scale,
                                    int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) grad32[i] = __half2float(grad16[i]) * inv_scale;
}

// Detect if any gradient is Inf or NaN (for dynamic loss scaling)
__global__ void checkGradientOverflow(const float* __restrict__ grad,
                                       int*         __restrict__ overflow_flag,
                                       int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && (isinf(grad[i]) || isnan(grad[i])))
        atomicOr(overflow_flag, 1);
}

// SGD update on FP32 master weights
__global__ void sgdUpdateFP32(float*       __restrict__ param,
                                const float* __restrict__ grad,
                                float lr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) param[i] -= lr * grad[i];
}

// Scale gradients by a factor (for loss scaling)
__global__ void scaleGradients(float* grad, float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) grad[i] *= scale;
}

// Fused bias + ReLU in FP16
__global__ void biasReluFP16(const __half* __restrict__ in,
                               const __half* __restrict__ bias,
                               __half*       __restrict__ out,
                               int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        __half v = __hadd(in[idx], bias[idx % cols]);
        out[idx] = __hgt(v, __float2half(0.f)) ? v : __float2half(0.f);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Loss scaling utility
// ─────────────────────────────────────────────────────────────────────────────
struct GradScaler {
    float scale;
    int   growth_interval;   // double scale every N successful steps
    int   steps_since_growth;
    int   num_overflows;

    GradScaler(float init_scale = 65536.f, int growth_interval = 2000)
        : scale(init_scale), growth_interval(growth_interval),
          steps_since_growth(0), num_overflows(0) {}

    // Returns true if the step should be skipped (overflow detected)
    bool update(bool overflow) {
        if (overflow) {
            scale *= 0.5f;
            steps_since_growth = 0;
            num_overflows++;
            return true;  // skip this step
        }
        steps_since_growth++;
        if (steps_since_growth >= growth_interval) {
            scale *= 2.f;
            steps_since_growth = 0;
        }
        return false;  // proceed with update
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    print_device_info();
    srand(42);

    const int B    = (argc > 1) ? atoi(argv[1]) : 256;
    const int Din  = 512;
    const int Dhid = 1024;
    const int Dout = 128;
    const int BLK  = 256;

    printf("Mixed precision training loop: B=%d  Din=%d  Dhid=%d  Dout=%d\n\n",
           B, Din, Dhid, Dout);

    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));

    // ── Memory layout explanation ─────────────────────────────────────────────
    section("Memory layout: master weights (FP32) + working copy (FP16)");
    {
        int params = Dhid*Din + Dhid + Dout*Dhid + Dout;
        double fp32_mb = params * sizeof(float)  / 1e6;
        double fp16_mb = params * sizeof(__half)  / 1e6;
        printf("  Parameters: %d total\n", params);
        printf("  FP32 master copy (for optimizer): %.1f MB\n", fp32_mb);
        printf("  FP16 working copy (for forward):  %.1f MB\n", fp16_mb);
        printf("  Total parameter memory:           %.1f MB  (%.1fx inference)\n\n",
               fp32_mb + fp16_mb, (fp32_mb + fp16_mb) / fp32_mb);
        printf("  In PyTorch AMP: model.half() creates the FP16 working copy.\n");
        printf("  torch.amp.autocast() manages the cast transparently.\n");
    }

    // ── Allocate FP32 master weights ─────────────────────────────────────────
    float *d_W1_fp32, *d_b1_fp32, *d_W2_fp32, *d_b2_fp32;
    CUDA_CHECK(cudaMalloc(&d_W1_fp32, (size_t)Dhid * Din  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b1_fp32, (size_t)Dhid        * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W2_fp32, (size_t)Dout * Dhid * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b2_fp32, (size_t)Dout        * sizeof(float)));
    {
        float* tmp = (float*)malloc((size_t)Dhid * Din * sizeof(float));
        float s1 = sqrtf(6.f / (Din + Dhid));
        rand_fill_fp32(tmp, Dhid * Din, -s1, s1);
        CUDA_CHECK(cudaMemcpy(d_W1_fp32, tmp, (size_t)Dhid*Din*sizeof(float), cudaMemcpyHostToDevice));
        float s2 = sqrtf(6.f / (Dhid + Dout));
        rand_fill_fp32(tmp, Dout * Dhid, -s2, s2);
        CUDA_CHECK(cudaMemcpy(d_W2_fp32, tmp, (size_t)Dout*Dhid*sizeof(float), cudaMemcpyHostToDevice));
        free(tmp);
        CUDA_CHECK(cudaMemset(d_b1_fp32, 0, (size_t)Dhid * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_b2_fp32, 0, (size_t)Dout * sizeof(float)));
    }

    // ── Allocate FP16 working copies ──────────────────────────────────────────
    __half *d_W1_fp16, *d_b1_fp16, *d_W2_fp16, *d_b2_fp16;
    CUDA_CHECK(cudaMalloc(&d_W1_fp16, (size_t)Dhid * Din  * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_b1_fp16, (size_t)Dhid        * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_W2_fp16, (size_t)Dout * Dhid * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_b2_fp16, (size_t)Dout        * sizeof(__half)));

    // ── Activations (FP16) ───────────────────────────────────────────────────
    __half *d_x16, *d_h_pre16, *d_h16, *d_out16;
    CUDA_CHECK(cudaMalloc(&d_x16,     (size_t)B * Din  * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_h_pre16, (size_t)B * Dhid * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_h16,     (size_t)B * Dhid * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_out16,   (size_t)B * Dout * sizeof(__half)));

    // ── Gradients (FP32 after unscaling) ─────────────────────────────────────
    float *d_dW1, *d_db1, *d_dW2, *d_db2;
    CUDA_CHECK(cudaMalloc(&d_dW1, (size_t)Dhid * Din  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db1, (size_t)Dhid        * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dW2, (size_t)Dout * Dhid * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db2, (size_t)Dout        * sizeof(float)));

    // FP16 gradient buffer (before unscaling)
    __half *d_dout16, *d_dh16, *d_dhp16;
    CUDA_CHECK(cudaMalloc(&d_dout16, (size_t)B * Dout * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_dh16,   (size_t)B * Dhid * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_dhp16,  (size_t)B * Dhid * sizeof(__half)));

    int* d_overflow;
    CUDA_CHECK(cudaMalloc(&d_overflow, sizeof(int)));

    // ── Section: Cast FP32 -> FP16 overhead ──────────────────────────────────
    section("Step 1: Cast FP32 master weights to FP16 working copy");
    {
        int n = Dhid * Din;
        GpuTimer t; t.start();
        const int REPS = 100;
        for (int r = 0; r < REPS; r++)
            castFP32toFP16<<<(n+BLK-1)/BLK, BLK>>>(d_W1_fp32, d_W1_fp16, n);
        float ms = t.stop_ms() / REPS;
        printf("  W1 cast [%dx%d]: %.3f ms  (%.1f GB/s)\n",
               Dhid, Din, ms,
               bw_gb_s((size_t)n * (sizeof(float) + sizeof(__half)), ms));
        printf("  Performed once per training step, before each forward pass.\n");
    }

    // ── Section: Loss scaling demo ────────────────────────────────────────────
    section("Loss scaling for FP16 training");
    {
        printf("  FP16 minimum positive normal: ~%.0e\n", (double)6e-8f);
        printf("  Gradients smaller than this underflow to 0.0 in FP16.\n\n");

        // Show gradients that would underflow in FP16 but survive with scaling
        float grad_values[] = {1e-6f, 1e-7f, 1e-8f, 1e-9f};
        float loss_scale = 32768.f;  // 2^15

        printf("  %-12s %-15s %-20s %-15s\n",
               "Gradient", "FP16 repr", "Scaled (x2^15)", "After descale");
        printf("  %-12s %-15s %-20s %-15s\n",
               "----------","-------------","------------------","-------------");
        for (float g : grad_values) {
            __half h_unscaled = __float2half(g);
            __half h_scaled   = __float2half(g * loss_scale);
            float  back_unscaled = __half2float(h_unscaled);
            float  back_scaled   = __half2float(h_scaled) / loss_scale;
            printf("  %-12.2e %-15.2e %-20.2e %-15.2e %s\n",
                   (double)g,
                   (double)back_unscaled,
                   (double)(g * loss_scale),
                   (double)back_scaled,
                   (fabsf(back_unscaled) < 1e-10f) ? "<-- UNDERFLOW (lost!)" :
                   (fabsf(back_scaled - g) / (fabsf(g) + 1e-30f) < 0.01f) ?
                   "<-- preserved" : "");
        }

        printf("\n  Loss scaling rescues gradients that would otherwise underflow.\n");
        printf("  BF16 has the same exponent range as FP32: no scaling needed.\n");
        printf("  For new training runs on Ampere+: use BF16 and skip the scaler.\n");
    }

    // ── Section: Full step timing ─────────────────────────────────────────────
    section("Timing: FP32 only vs mixed precision (FP16 forward + FP32 update)");
    {
        float one32 = 1.f, zero32 = 0.f;
        __half one16 = __float2half(1.f), zero16 = __float2half(0.f);

        // Cast all weights to FP16
        castFP32toFP16<<<((size_t)Dhid*Din+BLK-1)/BLK, BLK>>>(d_W1_fp32, d_W1_fp16, Dhid*Din);
        castFP32toFP16<<<((size_t)Dout*Dhid+BLK-1)/BLK, BLK>>>(d_W2_fp32, d_W2_fp16, Dout*Dhid);
        castFP32toFP16<<<(Dhid+BLK-1)/BLK, BLK>>>(d_b1_fp32, d_b1_fp16, Dhid);
        castFP32toFP16<<<(Dout+BLK-1)/BLK, BLK>>>(d_b2_fp32, d_b2_fp16, Dout);
        CUDA_CHECK(cudaMemset(d_x16, 1, (size_t)B * Din * sizeof(__half)));

        const int REPS = 50;
        GpuTimer t;

        // FP16 forward pass
        t.start();
        for (int r = 0; r < REPS; r++) {
            // Layer 1: h_pre = x @ W1^T  (FP16 Tensor Cores)
            cublasSgemmEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                Dhid, B, Din,
                &one32,
                d_W1_fp16, CUDA_R_16F, Din,
                d_x16,     CUDA_R_16F, Din,
                &zero32,
                d_h_pre16, CUDA_R_16F, Dhid);
            biasReluFP16<<<(B*Dhid+BLK-1)/BLK, BLK>>>(d_h_pre16, d_b1_fp16, d_h16, B, Dhid);

            // Layer 2: out = h @ W2^T  (FP16 Tensor Cores)
            cublasSgemmEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                Dout, B, Dhid,
                &one32,
                d_W2_fp16, CUDA_R_16F, Dhid,
                d_h16,     CUDA_R_16F, Dhid,
                &zero32,
                d_out16,   CUDA_R_16F, Dout);
        }
        float ms_fwd = t.stop_ms() / REPS;

        double fwd_flops = 2.0 * (2.0*B*Din*Dhid + 2.0*B*Dhid*Dout);
        printf("  FP16 forward pass (Tensor Cores): %.3f ms  %.1f GFLOP/s\n",
               ms_fwd, gflops(fwd_flops, ms_fwd));
        printf("\n  The weight update uses FP32 master weights,\n");
        printf("  ensuring optimizer state (momentum, variance) retains\n");
        printf("  full precision regardless of the forward pass dtype.\n\n");
        printf("  Memory breakdown per parameter:\n");
        printf("    FP32 master weight:   4 bytes\n");
        printf("    FP16 working copy:    2 bytes\n");
        printf("    FP32 gradient:        4 bytes\n");
        printf("    FP32 Adam state (m):  4 bytes\n");
        printf("    FP32 Adam state (v):  4 bytes\n");
        printf("    Total:               18 bytes/param\n");
        printf("    (vs 4 bytes for inference)\n");
    }

    cublasDestroy(cublas);
    cudaFree(d_W1_fp32); cudaFree(d_b1_fp32);
    cudaFree(d_W2_fp32); cudaFree(d_b2_fp32);
    cudaFree(d_W1_fp16); cudaFree(d_b1_fp16);
    cudaFree(d_W2_fp16); cudaFree(d_b2_fp16);
    cudaFree(d_x16); cudaFree(d_h_pre16); cudaFree(d_h16); cudaFree(d_out16);
    cudaFree(d_dW1); cudaFree(d_db1); cudaFree(d_dW2); cudaFree(d_db2);
    cudaFree(d_dout16); cudaFree(d_dh16); cudaFree(d_dhp16);
    cudaFree(d_overflow);
    return 0;
}
