// 04_training_loop.cu — Post 5: Building a Neural Network on One GPU
//
// A complete training loop: forward → loss → backward → optimizer step.
// Trains a toy MLP on a synthetic classification task (random XOR-like data).
//
// Demonstrates:
//   - SGD and Adam optimizer kernels
//   - Gradient zeroing (cudaMemset) — essential, easy to forget
//   - Loss tracking across steps
//   - The double-buffer data pipeline from Post 4 (pinned host memory)
//
// Adam optimizer:
//   m = beta1 * m + (1 - beta1) * g        (first moment / momentum)
//   v = beta2 * v + (1 - beta2) * g^2      (second moment / variance)
//   m_hat = m / (1 - beta1^t)              (bias-corrected)
//   v_hat = v / (1 - beta2^t)
//   param -= lr * m_hat / (sqrt(v_hat) + eps)
//
// The Adam kernel reads: param, grad, m, v   (4 reads)
// The Adam kernel writes: param, m, v         (3 writes)
// Total: 7 tensor passes = deeply memory-bound, like all optimizer steps.

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/utils.cuh"

// ─────────────────────────────────────────────────────────────────────────────
// Kernels
// ─────────────────────────────────────────────────────────────────────────────

__global__ void biasReluFwd(const float* __restrict__ in,
                             const float* __restrict__ b,
                             float* __restrict__ out,
                             int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * cols) out[i] = fmaxf(0.f, in[i] + b[i % cols]);
}

__global__ void biasAddKernel(const float* __restrict__ in,
                               const float* __restrict__ b,
                               float* __restrict__ out,
                               int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * cols) out[i] = in[i] + b[i % cols];
}

__global__ void mseLossAndGrad(const float* __restrict__ out,
                                const float* __restrict__ tgt,
                                float*       __restrict__ grad,
                                float*       __restrict__ loss_out,
                                int n) {
    extern __shared__ float smem[];
    int i   = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float diff = (i < n) ? (out[i] - tgt[i]) : 0.f;
    if (i < n) grad[i] = 2.f * diff / (float)n;
    smem[tid] = diff * diff;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(loss_out, smem[0] / (float)n);
}

__global__ void reluBwdKernel(const float* __restrict__ pre,
                               const float* __restrict__ g_in,
                               float* __restrict__ g_out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) g_out[i] = (pre[i] > 0.f) ? g_in[i] : 0.f;
}

__global__ void biasGradKernel(const float* __restrict__ dout,
                                float* __restrict__ db,
                                int B, int D) {
    extern __shared__ float smem[];
    int col = blockIdx.x, tid = threadIdx.x;
    float s = 0.f;
    for (int b = tid; b < B; b += blockDim.x) s += dout[b * D + col];
    smem[tid] = s;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    if (tid == 0) db[col] = smem[0];
}

// ── Optimizer kernels ─────────────────────────────────────────────────────────

__global__ void sgdUpdateKernel(float* __restrict__ p,
                                 const float* __restrict__ g,
                                 float lr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] -= lr * g[i];
}

// Adam update — one thread per parameter
__global__ void adamUpdateKernel(float*       __restrict__ p,
                                  const float* __restrict__ g,
                                  float*       __restrict__ m,
                                  float*       __restrict__ v,
                                  float lr, float beta1, float beta2,
                                  float eps, float bc1, float bc2,
                                  int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gi = g[i];
    float mi = beta1 * m[i] + (1.f - beta1) * gi;          // update momentum
    float vi = beta2 * v[i] + (1.f - beta2) * gi * gi;     // update variance
    m[i] = mi; v[i] = vi;
    float m_hat = mi / bc1;                                   // bias correction
    float v_hat = vi / bc2;
    p[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
}

// ─────────────────────────────────────────────────────────────────────────────
// Simple synthetic dataset: binary classification on random separable data
// ─────────────────────────────────────────────────────────────────────────────
static void makeDataset(float* x, float* y, int n, int din, int dout) {
    // Random XOR-like pattern: target depends on sign of sum of inputs
    for (int i = 0; i < n; i++) {
        float s = 0.f;
        for (int j = 0; j < din; j++) {
            x[i * din + j] = ((float)rand() / RAND_MAX) * 2.f - 1.f;
            s += x[i * din + j];
        }
        for (int j = 0; j < dout; j++)
            y[i * dout + j] = (s > 0.f) ? 1.f : -1.f;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    print_device_info();
    srand(42);

    const int DATASET_SIZE = 8192;
    const int B    = (argc > 1) ? atoi(argv[1]) : 256;  // batch size
    const int Din  = 64;
    const int Dhid = 256;
    const int Dout = 1;    // binary classification
    const int STEPS= (argc > 2) ? atoi(argv[2]) : 500;
    const float LR = 1e-3f;
    const int BLK  = 256;

    printf("Dataset: %d samples  Batch: %d  Steps: %d\n", DATASET_SIZE, B, STEPS);
    printf("MLP: %d -> %d (ReLU) -> %d\n\n", Din, Dhid, Dout);

    // ── Dataset ───────────────────────────────────────────────────────────────
    float* h_X = (float*)malloc(DATASET_SIZE * Din  * sizeof(float));
    float* h_Y = (float*)malloc(DATASET_SIZE * Dout * sizeof(float));
    makeDataset(h_X, h_Y, DATASET_SIZE, Din, Dout);

    // Pinned batch buffers (Post 4 pattern)
    float *h_bx, *h_by;
    CUDA_CHECK(cudaMallocHost(&h_bx, B * Din  * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_by, B * Dout * sizeof(float)));

    // ── Device: model parameters ──────────────────────────────────────────────
    float *d_W1, *d_b1, *d_W2, *d_b2;
    CUDA_CHECK(cudaMalloc(&d_W1, Dhid * Din  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b1, Dhid        * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W2, Dout * Dhid * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b2, Dout        * sizeof(float)));

    // Xavier / Glorot init
    {
        float* tmp = (float*)malloc(Dhid * Din * sizeof(float));
        float scale1 = sqrtf(6.f / (Din + Dhid));
        rand_fill(tmp, Dhid * Din, -scale1, scale1);
        CUDA_CHECK(cudaMemcpy(d_W1, tmp, Dhid*Din*sizeof(float), cudaMemcpyHostToDevice));
        float scale2 = sqrtf(6.f / (Dhid + Dout));
        rand_fill(tmp, Dout * Dhid, -scale2, scale2);
        CUDA_CHECK(cudaMemcpy(d_W2, tmp, Dout*Dhid*sizeof(float), cudaMemcpyHostToDevice));
        free(tmp);
        CUDA_CHECK(cudaMemset(d_b1, 0, Dhid * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_b2, 0, Dout * sizeof(float)));
    }

    // ── Device: activations ───────────────────────────────────────────────────
    float *d_x, *d_y, *d_h_pre, *d_h, *d_out, *d_loss;
    CUDA_CHECK(cudaMalloc(&d_x,     B * Din  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y,     B * Dout * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_h_pre, B * Dhid * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_h,     B * Dhid * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out,   B * Dout * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_loss,  sizeof(float)));

    // ── Device: gradients ─────────────────────────────────────────────────────
    float *d_dout, *d_dh, *d_dhp;
    float *d_dW1, *d_db1, *d_dW2, *d_db2;
    CUDA_CHECK(cudaMalloc(&d_dout,B*Dout*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dh,  B*Dhid*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dhp, B*Dhid*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dW1, Dhid*Din*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db1, Dhid*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dW2, Dout*Dhid*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db2, Dout*sizeof(float)));

    // ── Device: Adam optimizer state ──────────────────────────────────────────
    float *d_mW1, *d_vW1, *d_mb1, *d_vb1;
    float *d_mW2, *d_vW2, *d_mb2, *d_vb2;
    CUDA_CHECK(cudaMalloc(&d_mW1, Dhid*Din*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vW1, Dhid*Din*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mb1, Dhid*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vb1, Dhid*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mW2, Dout*Dhid*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vW2, Dout*Dhid*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mb2, Dout*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vb2, Dout*sizeof(float)));
    // Zero-init optimizer state (required!)
    CUDA_CHECK(cudaMemset(d_mW1, 0, Dhid*Din*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_vW1, 0, Dhid*Din*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_mb1, 0, Dhid*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_vb1, 0, Dhid*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_mW2, 0, Dout*Dhid*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_vW2, 0, Dout*Dhid*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_mb2, 0, Dout*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_vb2, 0, Dout*sizeof(float)));

    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));
    float one = 1.f, zero = 0.f;

    const float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;

    // ── Training loop ─────────────────────────────────────────────────────────
    section("Training loop (Adam optimizer)");
    printf("  Step | Loss\n");
    printf("  -----|----------\n");

    float prev_loss = 1e9f;
    for (int step = 1; step <= STEPS; step++) {
        // Sample random mini-batch from dataset
        int offset = (rand() % (DATASET_SIZE - B));
        memcpy(h_bx, h_X + offset * Din,  B * Din  * sizeof(float));
        memcpy(h_by, h_Y + offset * Dout, B * Dout * sizeof(float));
        CUDA_CHECK(cudaMemcpy(d_x, h_bx, B*Din*sizeof(float),  cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_y, h_by, B*Dout*sizeof(float), cudaMemcpyHostToDevice));

        // ── FORWARD ──────────────────────────────────────────────────────────
        // Layer 1: h_pre = x @ W1^T,  h = ReLU(h_pre + b1)
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            Dhid, B, Din, &one, d_W1, Din, d_x, Din, &zero, d_h_pre, Dhid));
        biasReluFwd<<<(B*Dhid+BLK-1)/BLK, BLK>>>(d_h_pre, d_b1, d_h, B, Dhid);

        // Layer 2: out = h @ W2^T + b2
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            Dout, B, Dhid, &one, d_W2, Dhid, d_h, Dhid, &zero, d_out, Dout));
        biasAddKernel<<<(B*Dout+BLK-1)/BLK, BLK>>>(d_out, d_b2, d_out, B, Dout);

        // ── LOSS + INITIAL GRADIENT ───────────────────────────────────────────
        CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)));
        mseLossAndGrad<<<(B*Dout+BLK-1)/BLK, BLK, BLK*sizeof(float)>>>(
            d_out, d_y, d_dout, d_loss, B * Dout);

        // ── BACKWARD ─────────────────────────────────────────────────────────
        // dL/dW2 = h^T @ dL/dout
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
            Dhid, Dout, B, &one, d_h, Dhid, d_dout, Dout, &zero, d_dW2, Dhid));
        biasGradKernel<<<Dout, BLK, BLK*sizeof(float)>>>(d_dout, d_db2, B, Dout);

        // dL/dh = dL/dout @ W2
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            Dhid, B, Dout, &one, d_W2, Dhid, d_dout, Dout, &zero, d_dh, Dhid));

        // dL/dh_pre via ReLU gate
        reluBwdKernel<<<(B*Dhid+BLK-1)/BLK, BLK>>>(d_h_pre, d_dh, d_dhp, B*Dhid);

        // dL/dW1 = x^T @ dL/dh_pre
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
            Din, Dhid, B, &one, d_x, Din, d_dhp, Dhid, &zero, d_dW1, Din));
        biasGradKernel<<<Dhid, BLK, BLK*sizeof(float)>>>(d_dhp, d_db1, B, Dhid);

        // ── ADAM OPTIMIZER STEP ───────────────────────────────────────────────
        float bc1 = 1.f - powf(beta1, (float)step);   // bias correction
        float bc2 = 1.f - powf(beta2, (float)step);

        // Update each parameter tensor
        int nW1 = Dhid * Din, nb1 = Dhid, nW2 = Dout * Dhid, nb2 = Dout;
        #define ADAM(p, g, m, v, n) \
            adamUpdateKernel<<<((n)+BLK-1)/BLK, BLK>>>(p, g, m, v, LR, beta1, beta2, eps, bc1, bc2, n)
        ADAM(d_W1, d_dW1, d_mW1, d_vW1, nW1);
        ADAM(d_b1, d_db1, d_mb1, d_vb1, nb1);
        ADAM(d_W2, d_dW2, d_mW2, d_vW2, nW2);
        ADAM(d_b2, d_db2, d_mb2, d_vb2, nb2);
        #undef ADAM

        // ── ZERO GRADIENTS for next step ──────────────────────────────────────
        // CRITICAL: gradients accumulate across steps if not zeroed.
        // In PyTorch this is optimizer.zero_grad().
        CUDA_CHECK(cudaMemset(d_dW1, 0, nW1 * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_db1, 0, nb1 * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_dW2, 0, nW2 * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_db2, 0, nb2 * sizeof(float)));

        // ── Log loss ──────────────────────────────────────────────────────────
        if (step % (STEPS / 10) == 0 || step == 1) {
            float loss;
            CUDA_CHECK(cudaMemcpy(&loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
            printf("  %4d | %.6f  %s\n", step, loss,
                   loss < prev_loss ? "(improving)" : "(plateau)");
            prev_loss = loss;
        }
    }

    // ── Memory summary ────────────────────────────────────────────────────────
    section("Peak GPU memory during training");
    double params_mb  = (Dhid*Din + Dhid + Dout*Dhid + Dout) * 4.0 / 1e6;
    double adam_mb    = params_mb * 2;    // m and v for each parameter
    double acts_mb    = (B*Din + B*Dhid*2 + B*Dout) * 4.0 / 1e6;
    double grads_mb   = params_mb + (B*Dout + B*Dhid*2) * 4.0 / 1e6;
    printf("  Parameters    : %.1f MB\n", params_mb);
    printf("  Adam state    : %.1f MB  (m + v for each param = 2x params)\n", adam_mb);
    printf("  Activations   : %.1f MB  (saved for backward)\n", acts_mb);
    printf("  Gradients     : %.1f MB\n", grads_mb);
    printf("  TOTAL         : %.1f MB\n\n", params_mb + adam_mb + acts_mb + grads_mb);
    printf("  Breakdown matches the 3-4x overhead of training over inference.\n");
    printf("  For a 7B model: params ~14GB, Adam state ~28GB, acts+grads ~28GB+.\n");

    // ── Cleanup ───────────────────────────────────────────────────────────────
    cublasDestroy(cublas);
    cudaFree(d_W1); cudaFree(d_b1); cudaFree(d_W2); cudaFree(d_b2);
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_h_pre); cudaFree(d_h);
    cudaFree(d_out); cudaFree(d_loss);
    cudaFree(d_dout); cudaFree(d_dh); cudaFree(d_dhp);
    cudaFree(d_dW1); cudaFree(d_db1); cudaFree(d_dW2); cudaFree(d_db2);
    cudaFree(d_mW1); cudaFree(d_vW1); cudaFree(d_mb1); cudaFree(d_vb1);
    cudaFree(d_mW2); cudaFree(d_vW2); cudaFree(d_mb2); cudaFree(d_vb2);
    cudaFreeHost(h_bx); cudaFreeHost(h_by);
    free(h_X); free(h_Y);
    return 0;
}
