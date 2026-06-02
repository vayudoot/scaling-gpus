// 03_backprop.cu — Post 5: Building a Neural Network on One GPU
//
// Backpropagation from scratch: every gradient is a matmul or elementwise op.
//
// Forward:
//   h_pre = x @ W1^T + b1         [B x Dhid]  <- SAVED (ReLU backward needs it)
//   h     = ReLU(h_pre)           [B x Dhid]  <- SAVED (layer2 backward needs it)
//   out   = h @ W2^T + b2         [B x Dout]
//   loss  = MSE(out, target)
//
// Backward (chain rule):
//   dL/d_out  = 2*(out-target)/N  [B x Dout]
//   dL/dW2    = h^T @ dL/d_out   [Dhid x Dout]
//   dL/db2    = sum_rows(dL/dout) [Dout]
//   dL/dh     = dL/d_out @ W2    [B x Dhid]
//   dL/dh_pre = dL/dh * (h_pre>0)[B x Dhid]   (ReLU gate)
//   dL/dW1    = x^T @ dL/dh_pre  [Din x Dhid]
//   dL/db1    = sum_rows(dL/dh_pre) [Dhid]
//
// Each arrow in the backward graph is either:
//   - A cuBLAS SGEMM (for weight and input gradients)
//   - A parallel reduction (for bias gradients)
//   - An elementwise kernel (for activation gradients)

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/utils.cuh"

// ─────────────────────────────────────────────────────────────────────────────
// Kernels
// ─────────────────────────────────────────────────────────────────────────────

// Fused bias + ReLU: out[i] = max(0, in[i] + bias[i % cols])
__global__ void biasReluFwd(const float* __restrict__ in,
                             const float* __restrict__ bias,
                             float*       __restrict__ out,
                             int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols)
        out[idx] = fmaxf(0.f, in[idx] + bias[idx % cols]);
}

// Bias add (no activation): out[i] = in[i] + bias[i % cols]
__global__ void biasAdd(const float* __restrict__ in,
                         const float* __restrict__ bias,
                         float*       __restrict__ out,
                         int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols)
        out[idx] = in[idx] + bias[idx % cols];
}

// MSE loss gradient: dL/d_out = 2*(out - target) / n
__global__ void mseLossGrad(const float* __restrict__ out,
                             const float* __restrict__ tgt,
                             float*       __restrict__ grad,
                             int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) grad[i] = 2.f * (out[i] - tgt[i]) / (float)n;
}

// ReLU backward: pass gradient where forward input was positive
__global__ void reluBwd(const float* __restrict__ h_pre,   // saved forward input
                         const float* __restrict__ grad_in,
                         float*       __restrict__ grad_out,
                         int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) grad_out[i] = (h_pre[i] > 0.f) ? grad_in[i] : 0.f;
}

// Bias gradient: sum over the batch dimension
// Each block handles one output column; threads reduce over rows.
__global__ void biasGrad(const float* __restrict__ dL_dout,  // [B x D]
                          float*       __restrict__ dL_db,    // [D]
                          int B, int D) {
    extern __shared__ float smem[];
    int col = blockIdx.x;
    int tid = threadIdx.x;
    float s = 0.f;
    for (int b = tid; b < B; b += blockDim.x)
        s += dL_dout[b * D + col];
    smem[tid] = s;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    if (tid == 0) dL_db[col] = smem[0];
}

// SGD weight update: param -= lr * grad
__global__ void sgdUpdate(float*       __restrict__ param,
                           const float* __restrict__ grad,
                           float lr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) param[i] -= lr * grad[i];
}

// ─────────────────────────────────────────────────────────────────────────────
// CPU reference (small batches only)
// ─────────────────────────────────────────────────────────────────────────────
static void cpuBackward(const float* x, const float* h_pre, const float* h,
                         const float* W2, const float* dL_dout,
                         float* dL_dW1, float* dL_db1,
                         float* dL_dW2, float* dL_db2,
                         int B, int Din, int Dhid, int Dout) {
    // dL/dW2 += h^T @ dL/dout
    memset(dL_dW2, 0, Dout * Dhid * sizeof(float));
    for (int b = 0; b < B; b++)
        for (int j = 0; j < Dout; j++)
            for (int i = 0; i < Dhid; i++)
                dL_dW2[j * Dhid + i] += h[b*Dhid+i] * dL_dout[b*Dout+j];
    // dL/db2 = sum_rows(dL/dout)
    memset(dL_db2, 0, Dout * sizeof(float));
    for (int b = 0; b < B; b++)
        for (int j = 0; j < Dout; j++)
            dL_db2[j] += dL_dout[b*Dout+j];
    // dL/dh = dL/dout @ W2
    float* dL_dh = (float*)calloc((size_t)B * Dhid, sizeof(float));
    for (int b = 0; b < B; b++)
        for (int i = 0; i < Dhid; i++)
            for (int j = 0; j < Dout; j++)
                dL_dh[b*Dhid+i] += dL_dout[b*Dout+j] * W2[j*Dhid+i];
    // dL/dh_pre = dL/dh * (h_pre > 0)
    float* dL_dhp = (float*)malloc((size_t)B * Dhid * sizeof(float));
    for (int i = 0; i < B*Dhid; i++)
        dL_dhp[i] = (h_pre[i] > 0.f) ? dL_dh[i] : 0.f;
    // dL/dW1 += x^T @ dL/dh_pre
    memset(dL_dW1, 0, Dhid * Din * sizeof(float));
    for (int b = 0; b < B; b++)
        for (int j = 0; j < Dhid; j++)
            for (int i = 0; i < Din; i++)
                dL_dW1[j*Din+i] += x[b*Din+i] * dL_dhp[b*Dhid+j];
    // dL/db1 = sum_rows(dL/dh_pre)
    memset(dL_db1, 0, Dhid * sizeof(float));
    for (int b = 0; b < B; b++)
        for (int j = 0; j < Dhid; j++)
            dL_db1[j] += dL_dhp[b*Dhid+j];
    free(dL_dh); free(dL_dhp);
}

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

    printf("MLP backward: B=%d  Din=%d  Dhid=%d  Dout=%d\n\n", B, Din, Dhid, Dout);

    // ── Host allocs ──────────────────────────────────────────────────────────
    float* h_x   = (float*)malloc((size_t)B * Din * sizeof(float));
    float* h_W1  = (float*)malloc((size_t)Dhid * Din * sizeof(float));
    float* h_b1  = (float*)calloc(Dhid, sizeof(float));
    float* h_W2  = (float*)malloc((size_t)Dout * Dhid * sizeof(float));
    float* h_b2  = (float*)calloc(Dout, sizeof(float));
    float* h_tgt = (float*)malloc((size_t)B * Dout * sizeof(float));
    rand_fill(h_x,  B*Din,    -0.5f, 0.5f);
    rand_fill(h_W1, Dhid*Din,  -0.05f, 0.05f);
    rand_fill(h_W2, Dout*Dhid, -0.05f, 0.05f);
    rand_fill(h_tgt, B*Dout,  -1.f, 1.f);

    // ── Device allocs ─────────────────────────────────────────────────────────
    float *d_x, *d_W1, *d_b1, *d_W2, *d_b2, *d_tgt;
    float *d_h_pre, *d_h, *d_out;
    float *d_dout, *d_dh, *d_dhp, *d_dW1, *d_db1, *d_dW2, *d_db2;

    CUDA_CHECK(cudaMalloc(&d_x,    B*Din*4));    CUDA_CHECK(cudaMalloc(&d_W1,   Dhid*Din*4));
    CUDA_CHECK(cudaMalloc(&d_b1,   Dhid*4));     CUDA_CHECK(cudaMalloc(&d_W2,   Dout*Dhid*4));
    CUDA_CHECK(cudaMalloc(&d_b2,   Dout*4));     CUDA_CHECK(cudaMalloc(&d_tgt,  B*Dout*4));
    CUDA_CHECK(cudaMalloc(&d_h_pre,B*Dhid*4));   CUDA_CHECK(cudaMalloc(&d_h,    B*Dhid*4));
    CUDA_CHECK(cudaMalloc(&d_out,  B*Dout*4));
    CUDA_CHECK(cudaMalloc(&d_dout, B*Dout*4));   CUDA_CHECK(cudaMalloc(&d_dh,   B*Dhid*4));
    CUDA_CHECK(cudaMalloc(&d_dhp,  B*Dhid*4));   CUDA_CHECK(cudaMalloc(&d_dW1,  Dhid*Din*4));
    CUDA_CHECK(cudaMalloc(&d_db1,  Dhid*4));     CUDA_CHECK(cudaMalloc(&d_dW2,  Dout*Dhid*4));
    CUDA_CHECK(cudaMalloc(&d_db2,  Dout*4));

    CUDA_CHECK(cudaMemcpy(d_x,  h_x,  B*Din*4,    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W1, h_W1, Dhid*Din*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b1, h_b1, Dhid*4,     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W2, h_W2, Dout*Dhid*4,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2, h_b2, Dout*4,     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tgt,h_tgt,B*Dout*4,   cudaMemcpyHostToDevice));

    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));
    float one = 1.f, zero = 0.f;

    // ── Forward pass ─────────────────────────────────────────────────────────
    auto fwd = [&]() {
        // h_pre = x @ W1^T   [B x Dhid]
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            Dhid, B, Din, &one, d_W1, Din, d_x, Din, &zero, d_h_pre, Dhid));
        // h = ReLU(h_pre + b1)
        biasReluFwd<<<(B*Dhid+BLK-1)/BLK, BLK>>>(d_h_pre, d_b1, d_h, B, Dhid);
        // out = h @ W2^T + b2   [B x Dout]
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            Dout, B, Dhid, &one, d_W2, Dhid, d_h, Dhid, &zero, d_out, Dout));
        biasAdd<<<(B*Dout+BLK-1)/BLK, BLK>>>(d_out, d_b2, d_out, B, Dout);
    };

    // ── Backward pass ─────────────────────────────────────────────────────────
    auto bwd = [&]() {
        // Step 1: loss gradient
        mseLossGrad<<<(B*Dout+BLK-1)/BLK, BLK>>>(d_out, d_tgt, d_dout, B*Dout);

        // Step 2: dL/dW2 = h^T @ dL/dout   [Dhid x Dout]
        // In row-major: h is [B x Dhid], dL_dout is [B x Dout]
        // cuBLAS (col-major): dL_dout^T [Dout x B] @ h [B x Dhid] -> dW2^T [Dout x Dhid]
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
            Dhid, Dout, B, &one, d_h, Dhid, d_dout, Dout, &zero, d_dW2, Dhid));

        // Step 3: dL/db2 = sum over batch
        biasGrad<<<Dout, BLK, BLK*sizeof(float)>>>(d_dout, d_db2, B, Dout);

        // Step 4: dL/dh = dL/dout @ W2   [B x Dhid]
        // cuBLAS: W2^T [Dhid x Dout] @ dL_dout^T [Dout x B] -> dh^T [Dhid x B]
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            Dhid, B, Dout, &one, d_W2, Dhid, d_dout, Dout, &zero, d_dh, Dhid));

        // Step 5: ReLU backward — gate with saved h_pre
        reluBwd<<<(B*Dhid+BLK-1)/BLK, BLK>>>(d_h_pre, d_dh, d_dhp, B*Dhid);

        // Step 6: dL/dW1 = x^T @ dL/dh_pre   [Dhid x Din] (stored as [Dhid x Din])
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
            Din, Dhid, B, &one, d_x, Din, d_dhp, Dhid, &zero, d_dW1, Din));

        // Step 7: dL/db1 = sum over batch
        biasGrad<<<Dhid, BLK, BLK*sizeof(float)>>>(d_dhp, d_db1, B, Dhid);
    };

    // ── Warm up ───────────────────────────────────────────────────────────────
    fwd(); bwd(); CUDA_CHECK(cudaDeviceSynchronize());

    // ── Timing ────────────────────────────────────────────────────────────────
    section("Forward and backward pass timing");
    const int REPS = 100;
    GpuTimer t;

    t.start();
    for (int r = 0; r < REPS; r++) fwd();
    float ms_fwd = t.stop_ms() / REPS;

    t.start();
    for (int r = 0; r < REPS; r++) bwd();
    float ms_bwd = t.stop_ms() / REPS;

    double fwd_flops = 2.0 * (2.0*B*Din*Dhid + 2.0*B*Dhid*Dout);
    // Backward: 2 matmuls per layer (dL/dW and dL/dX)
    double bwd_flops = 4.0 * (2.0*B*Din*Dhid + 2.0*B*Dhid*Dout);

    printf("  Forward  : %.3f ms  (%.1f GFLOP/s)\n",
           ms_fwd, fwd_flops / (ms_fwd * 1e-3) / 1e9);
    printf("  Backward : %.3f ms  (%.1f GFLOP/s)\n",
           ms_bwd, bwd_flops / (ms_bwd * 1e-3) / 1e9);
    printf("  Bwd/Fwd  : %.1fx  (rule of thumb: ~2x)\n\n", ms_bwd / ms_fwd);

    // ── Memory cost breakdown ─────────────────────────────────────────────────
    section("Memory breakdown during one training step");
    double params  = (Dhid*Din + Dhid + Dout*Dhid + Dout) * 4.0 / 1e6;
    double acts    = (B*Din + B*Dhid*2 + B*Dout) * 4.0 / 1e6;
    double grads_w = params;
    double grads_a = (B*Dout + B*Dhid*2) * 4.0 / 1e6;
    printf("  Parameters        : %6.1f MB\n", params);
    printf("  Saved activations : %6.1f MB  (x, h_pre, h — needed for backward)\n", acts);
    printf("  Weight gradients  : %6.1f MB  (dW1, db1, dW2, db2)\n", grads_w);
    printf("  Activation grads  : %6.1f MB  (dout, dh, dhp)\n", grads_a);
    printf("  TOTAL training    : %6.1f MB\n", params+acts+grads_w+grads_a);
    printf("  Inference only    : %6.1f MB  (%.1fx less)\n\n",
           params, (params+acts+grads_w+grads_a)/params);

    // ── Correctness check ─────────────────────────────────────────────────────
    if (B <= 32) {
        section("Correctness check (small batch)");
        fwd(); CUDA_CHECK(cudaDeviceSynchronize());

        // Get dL/dout from GPU, compute CPU reference backward
        float* h_dout = (float*)malloc((size_t)B * Dout * 4);
        CUDA_CHECK(cudaMemcpy(h_dout, d_dout, B*Dout*4, cudaMemcpyDeviceToHost));

        // Need h_pre and h from GPU
        float* h_h_pre = (float*)malloc((size_t)B * Dhid * 4);
        float* h_h     = (float*)malloc((size_t)B * Dhid * 4);
        CUDA_CHECK(cudaMemcpy(h_h_pre, d_h_pre, B*Dhid*4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_h,     d_h,     B*Dhid*4, cudaMemcpyDeviceToHost));

        float* ref_dW1 = (float*)calloc((size_t)Dhid * Din, 4);
        float* ref_db1 = (float*)calloc(Dhid, 4);
        float* ref_dW2 = (float*)calloc((size_t)Dout * Dhid, 4);
        float* ref_db2 = (float*)calloc(Dout, 4);

        cpuBackward(h_x, h_h_pre, h_h, h_W2, h_dout,
                    ref_dW1, ref_db1, ref_dW2, ref_db2,
                    B, Din, Dhid, Dout);

        float* gpu_dW2 = (float*)malloc((size_t)Dout * Dhid * 4);
        CUDA_CHECK(cudaMemcpy(gpu_dW2, d_dW2, Dout*Dhid*4, cudaMemcpyDeviceToHost));
        printf("  dL/dW2 max abs error: %.2e\n",
               max_abs_diff(gpu_dW2, ref_dW2, Dout*Dhid));

        free(h_dout); free(h_h_pre); free(h_h);
        free(ref_dW1); free(ref_db1); free(ref_dW2); free(ref_db2); free(gpu_dW2);
    }

    cublasDestroy(cublas);
    cudaFree(d_x);  cudaFree(d_W1); cudaFree(d_b1);
    cudaFree(d_W2); cudaFree(d_b2); cudaFree(d_tgt);
    cudaFree(d_h_pre); cudaFree(d_h); cudaFree(d_out);
    cudaFree(d_dout); cudaFree(d_dh); cudaFree(d_dhp);
    cudaFree(d_dW1); cudaFree(d_db1); cudaFree(d_dW2); cudaFree(d_db2);
    free(h_x); free(h_W1); free(h_b1); free(h_W2); free(h_b2); free(h_tgt);
    return 0;
}
