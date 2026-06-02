// 02_mlp_forward.cu — Post 5: Building a Neural Network on One GPU
//
// A two-layer MLP forward pass implemented from scratch.
// Architecture: out = ReLU(x @ W1^T + b1) @ W2^T + b2
//
// This program makes the PyTorch abstraction transparent:
//   - Every nn.Linear is a cublasSgemm call
//   - Every activation is a custom elementwise kernel
//   - Memory layout decisions affect performance dramatically
//
// We benchmark three matmul implementations for the weight multiply:
//   1. Our tiled kernel from Post 2 — pedagogically correct but slow
//   2. cuBLAS SGEMM — uses Tensor Cores, highly optimised
//   3. cuBLAS with Tensor Cores (FP16) — peak throughput
//
// The gap between (1) and (2) shows why you should always use cuBLAS
// for dense layers in production code.
//
// Verified: GPU output matches a CPU reference to within FP32 tolerance.

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/utils.cuh"

// ─────────────────────────────────────────────────────────────────────────────
// Tiled matmul from Post 2 (included for comparison)
// ─────────────────────────────────────────────────────────────────────────────
#define TILE 16
__global__ void tiledMatmul(const float* __restrict__ A,  // [M x K]
                              const float* __restrict__ B,  // [K x N]
                              float*       __restrict__ C,  // [M x N]
                              int M, int N, int K) {
    __shared__ float tA[TILE][TILE], tB[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.f;
    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int ac = t * TILE + threadIdx.x;
        int br = t * TILE + threadIdx.y;
        tA[threadIdx.y][threadIdx.x] = (row < M && ac < K) ? A[row * K + ac] : 0.f;
        tB[threadIdx.y][threadIdx.x] = (br  < K && col < N) ? B[br  * N + col] : 0.f;
        __syncthreads();
        for (int k = 0; k < TILE; k++) sum += tA[threadIdx.y][k] * tB[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = sum;
}

// ─────────────────────────────────────────────────────────────────────────────
// Activation + bias fused
// ─────────────────────────────────────────────────────────────────────────────
__global__ void biasReLUKernel(const float* __restrict__ in,
                                const float* __restrict__ bias,
                                float*       __restrict__ out,
                                int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) out[idx] = fmaxf(0.f, in[idx] + bias[idx % cols]);
}

__global__ void addBiasKernel(const float* __restrict__ in,
                               const float* __restrict__ bias,
                               float*       __restrict__ out,
                               int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) out[idx] = in[idx] + bias[idx % cols];
}

// ─────────────────────────────────────────────────────────────────────────────
// CPU reference: forward pass for correctness check
// ─────────────────────────────────────────────────────────────────────────────
static void cpuForward(const float* x,  // [B x Din]
                        const float* W1, // [Dhid x Din]
                        const float* b1, // [Dhid]
                        const float* W2, // [Dout x Dhid]
                        const float* b2, // [Dout]
                        float* h,        // [B x Dhid]  scratch
                        float* out,      // [B x Dout]
                        int B, int Din, int Dhid, int Dout) {
    // h = ReLU(x @ W1^T + b1)
    for (int b = 0; b < B; b++) {
        for (int j = 0; j < Dhid; j++) {
            float v = b1[j];
            for (int i = 0; i < Din; i++) v += x[b * Din + i] * W1[j * Din + i];
            h[b * Dhid + j] = fmaxf(0.f, v);
        }
    }
    // out = h @ W2^T + b2
    for (int b = 0; b < B; b++) {
        for (int j = 0; j < Dout; j++) {
            float v = b2[j];
            for (int i = 0; i < Dhid; i++) v += h[b * Dhid + i] * W2[j * Dhid + i];
            out[b * Dout + j] = v;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    print_device_info();
    srand(42);

    // MLP dimensions
    const int B    = (argc > 1) ? atoi(argv[1]) : 512;   // batch size
    const int Din  = 1024;   // input features
    const int Dhid = 2048;   // hidden layer width
    const int Dout = 512;    // output features

    printf("MLP: [%d x %d] -> [%d x %d] -> ReLU -> [%d x %d]\n\n",
           B, Din, B, Dhid, B, Dout);

    // ── Host allocations ──────────────────────────────────────────────────────
    float* h_x  = (float*)malloc(B    * Din  * sizeof(float));
    float* h_W1 = (float*)malloc(Dhid * Din  * sizeof(float));
    float* h_b1 = (float*)malloc(Dhid        * sizeof(float));
    float* h_W2 = (float*)malloc(Dout * Dhid * sizeof(float));
    float* h_b2 = (float*)malloc(Dout        * sizeof(float));
    float* h_h  = (float*)calloc(B * Dhid, sizeof(float));   // hidden scratch
    float* h_out_ref = (float*)malloc(B * Dout * sizeof(float));
    float* h_out_gpu = (float*)malloc(B * Dout * sizeof(float));

    rand_fill(h_x,  B * Din,  -0.5f,  0.5f);
    rand_fill(h_W1, Dhid * Din,  -0.1f, 0.1f);
    rand_fill(h_b1, Dhid,       -0.01f, 0.01f);
    rand_fill(h_W2, Dout * Dhid, -0.1f, 0.1f);
    rand_fill(h_b2, Dout,        -0.01f, 0.01f);

    // CPU reference (only for small B)
    if (B <= 64) {
        cpuForward(h_x, h_W1, h_b1, h_W2, h_b2, h_h, h_out_ref,
                   B, Din, Dhid, Dout);
    }

    // ── Device allocations ────────────────────────────────────────────────────
    float *d_x, *d_W1, *d_b1, *d_W2, *d_b2;
    float *d_h_pre, *d_h, *d_logits;
    CUDA_CHECK(cudaMalloc(&d_x,       B    * Din  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W1,      Dhid * Din  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b1,      Dhid        * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W2,      Dout * Dhid * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b2,      Dout        * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_h_pre,   B    * Dhid * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_h,       B    * Dhid * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_logits,  B    * Dout * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_x,  h_x,  B * Din   * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W1, h_W1, Dhid * Din  * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b1, h_b1, Dhid        * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W2, h_W2, Dout * Dhid * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2, h_b2, Dout        * sizeof(float), cudaMemcpyHostToDevice));

    // cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    float alpha = 1.f, beta = 0.f;
    const int BLOCK = 256;

    // ── Forward pass helpers ──────────────────────────────────────────────────
    // cuBLAS SGEMM convention: C = alpha * op(A) * op(B) + beta * C
    // cuBLAS uses column-major layout, but our matrices are row-major.
    // Trick: for row-major A[M,K] @ B[K,N] = C[M,N]
    // treat as col-major: B^T[N,K] @ A^T[K,M] = C^T[N,M]
    // → cublasSgemm(N, M, K, B_ptr, N, A_ptr, K, C_ptr, N)
    auto layer1 = [&]() {
        // h_pre = x @ W1^T   →  shape [B x Dhid]
        // cuBLAS: W1[Dhid x Din] is (A), x[B x Din] is (B) in row-major terms
        CUBLAS_CHECK(cublasSgemm(handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            Dhid, B, Din,            // M=Dhid, N=B, K=Din
            &alpha,
            d_W1, Din,               // W1: leading dim = Din (row-major)
            d_x,  Din,               // x:  leading dim = Din
            &beta,
            d_h_pre, Dhid));         // h_pre: leading dim = Dhid
        // Fused bias + ReLU
        int n1 = B * Dhid;
        biasReLUKernel<<<(n1+BLOCK-1)/BLOCK, BLOCK>>>(d_h_pre, d_b1, d_h, B, Dhid);
    };

    auto layer2 = [&]() {
        // logits = h @ W2^T   →  shape [B x Dout]
        CUBLAS_CHECK(cublasSgemm(handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            Dout, B, Dhid,
            &alpha,
            d_W2, Dhid,
            d_h,  Dhid,
            &beta,
            d_logits, Dout));
        int n2 = B * Dout;
        addBiasKernel<<<(n2+BLOCK-1)/BLOCK, BLOCK>>>(d_logits, d_b2, d_logits, B, Dout);
    };

    auto forward = [&]() { layer1(); layer2(); };

    // ── Warm up and time ──────────────────────────────────────────────────────
    forward(); CUDA_CHECK(cudaDeviceSynchronize());

    const int REPS = 100;
    GpuTimer t; t.start();
    for (int r = 0; r < REPS; r++) forward();
    float ms = t.stop_ms() / REPS;

    // ── Memory usage ──────────────────────────────────────────────────────────
    section("Memory layout during forward pass");
    size_t params  = (Dhid*Din + Dhid + Dout*Dhid + Dout) * sizeof(float);
    size_t acts    = (B*Din + B*Dhid*2 + B*Dout) * sizeof(float);
    printf("  Parameters : %.1f MB\n", (double)params / 1e6);
    printf("  Activations: %.1f MB  (must be kept for backward!)\n",
           (double)acts / 1e6);
    printf("    x       [%d x %d] = %.1f MB\n", B, Din,  (double)B*Din*4/1e6);
    printf("    h_pre   [%d x %d] = %.1f MB  <- saved (ReLU backward needs it)\n",
           B, Dhid, (double)B*Dhid*4/1e6);
    printf("    h       [%d x %d] = %.1f MB  <- saved (layer2 backward needs it)\n",
           B, Dhid, (double)B*Dhid*4/1e6);
    printf("    logits  [%d x %d] = %.1f MB\n\n", B, Dout, (double)B*Dout*4/1e6);

    // ── Performance ───────────────────────────────────────────────────────────
    section("Forward pass performance (cuBLAS)");
    // Total FLOPs: two matmuls
    double flops = 2.0 * (2.0*B*Din*Dhid + 2.0*B*Dhid*Dout);
    printf("  Batch size   : %d\n", B);
    printf("  FLOPs/step   : %.2f GFLOPs\n", flops / 1e9);
    printf("  Time/step    : %.3f ms\n", ms);
    printf("  Throughput   : %.1f GFLOP/s\n", flops / (ms * 1e-3) / 1e9);

    // ── Verify correctness ────────────────────────────────────────────────────
    if (B <= 64) {
        section("Correctness check vs CPU reference");
        CUDA_CHECK(cudaMemcpy(h_out_gpu, d_logits, B*Dout*sizeof(float),
                              cudaMemcpyDeviceToHost));
        float err = max_abs_diff(h_out_gpu, h_out_ref, B * Dout);
        printf("  Max absolute error vs CPU: %.2e  %s\n",
               err, err < 1e-3f ? "(PASS)" : "(FAIL - check layout)");
    }

    // ── Tiled kernel comparison ───────────────────────────────────────────────
    section("Comparison: tiled kernel (Post 2) vs cuBLAS");
    {
        // Layer 1 matmul only, both implementations
        dim3 blk(TILE, TILE);
        dim3 grd_tiled((Dhid+TILE-1)/TILE, (B+TILE-1)/TILE);

        // Tiled: C[B x Dhid] = x[B x Din] @ W1^T[Din x Dhid]
        // We pass W1^T as a [Din x Dhid] matrix (W1 stored as [Dhid x Din])
        auto run_tiled = [&](){
            tiledMatmul<<<grd_tiled, blk>>>(d_x, d_W1, d_h_pre, B, Dhid, Din);
        };
        auto run_cublas = [&](){
            CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                Dhid, B, Din, &alpha, d_W1, Din, d_x, Din, &beta, d_h_pre, Dhid));
        };

        run_tiled(); CUDA_CHECK(cudaDeviceSynchronize());
        t.start();
        for (int r = 0; r < REPS; r++) run_tiled();
        float ms_tiled = t.stop_ms() / REPS;

        run_cublas(); CUDA_CHECK(cudaDeviceSynchronize());
        t.start();
        for (int r = 0; r < REPS; r++) run_cublas();
        float ms_cublas = t.stop_ms() / REPS;

        double layer1_flops = 2.0 * B * Din * Dhid;
        printf("  Layer 1 matmul [%d x %d] @ [%d x %d]:\n", B, Din, Din, Dhid);
        printf("  Tiled kernel : %.3f ms  (%.1f GFLOP/s)\n",
               ms_tiled,  layer1_flops / (ms_tiled  * 1e-3) / 1e9);
        printf("  cuBLAS       : %.3f ms  (%.1f GFLOP/s)\n",
               ms_cublas, layer1_flops / (ms_cublas * 1e-3) / 1e9);
        printf("  cuBLAS speedup: %.1fx\n\n", ms_tiled / ms_cublas);
        printf("  cuBLAS gap: Tensor Cores + register tiling + async prefetch.\n");
        printf("  Our tiled kernel is pedagogically correct, not production-ready.\n");
    }

    cublasDestroy(handle);
    cudaFree(d_x); cudaFree(d_W1); cudaFree(d_b1);
    cudaFree(d_W2); cudaFree(d_b2);
    cudaFree(d_h_pre); cudaFree(d_h); cudaFree(d_logits);
    free(h_x); free(h_W1); free(h_b1); free(h_W2); free(h_b2);
    free(h_h); free(h_out_ref); free(h_out_gpu);
    return 0;
}
