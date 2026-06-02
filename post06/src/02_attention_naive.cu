// 02_attention_naive.cu  —  Post 6: Attention on One GPU
//
// Full naive self-attention for one head:
//   Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V
//
// Implemented as four explicit steps, each writing a full tensor to HBM:
//   Step 1:  S = Q @ K^T           [N x N]  — the attention score matrix
//   Step 2:  S = S / sqrt(d)       [N x N]  — scaling
//   Step 3:  P = softmax(S)        [N x N]  — row-wise normalisation
//   Step 4:  O = P @ V             [N x d]  — weighted sum of values
//
// This program:
//   - Measures the time and bandwidth of each step
//   - Calculates arithmetic intensity and shows where each step sits on
//     the roofline (memory-bound vs compute-bound)
//   - Shows how memory usage grows quadratically with sequence length
//   - Verifies correctness against a CPU reference
//   - Runs at multiple sequence lengths to show the N^2 scaling
//
// The key observation: the N x N attention matrix at N=2048 costs
// 4 * 2048^2 bytes = 16 MB per head. At N=8192 that is 256 MB per head.
// With 32 heads the forward pass alone requires 8 GB for the attention
// matrices — before any activations, before any other layers.

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "../include/utils.cuh"

// ─────────────────────────────────────────────────────────────────────────────
// Kernels
// ─────────────────────────────────────────────────────────────────────────────

// Scale all elements by a scalar (used for 1/sqrt(d) scaling of attention scores)
__global__ void scaleKernel(float* data, float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] *= scale;
}

// Numerically stable row-wise softmax with causal mask option
__global__ void softmaxCausal(float* S,      // [N x N] in-place
                                int N,
                                bool causal) {  // if true, mask upper triangle
    int row = blockIdx.x;
    int tid = threadIdx.x;
    extern __shared__ float smem[];
    float* smem_max = smem;
    float* smem_sum = smem + blockDim.x;

    // How many columns this row can see (causal: only up to and including 'row')
    int cols = causal ? (row + 1) : N;

    // Pass 1: find row max (masked positions treated as -inf)
    float local_max = -FLT_MAX;
    for (int c = tid; c < N; c += blockDim.x) {
        float val = (c < cols) ? S[row * N + c] : -FLT_MAX;
        local_max = fmaxf(local_max, val);
    }
    smem_max[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) smem_max[tid] = fmaxf(smem_max[tid], smem_max[tid+s]);
        __syncthreads();
    }
    float row_max = smem_max[0];
    __syncthreads();

    // Pass 2: shifted exp and sum
    float local_sum = 0.f;
    for (int c = tid; c < N; c += blockDim.x) {
        float val = (c < cols) ? expf(S[row * N + c] - row_max) : 0.f;
        S[row * N + c] = val;   // store shifted exp in-place
        local_sum += val;
    }
    smem_sum[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) smem_sum[tid] += smem_sum[tid+s];
        __syncthreads();
    }
    float total = smem_sum[0];

    // Pass 3: normalise in-place
    for (int c = tid; c < N; c += blockDim.x)
        S[row * N + c] /= total;
}

// ─────────────────────────────────────────────────────────────────────────────
// CPU reference attention (for correctness check, small N only)
// ─────────────────────────────────────────────────────────────────────────────
static void cpuAttention(const float* Q, const float* K, const float* V,
                          float* O, int N, int d, bool causal) {
    float scale = 1.f / sqrtf((float)d);
    float* S = (float*)malloc((size_t)N * N * sizeof(float));

    // S = Q @ K^T * scale
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float dot = 0.f;
            for (int k = 0; k < d; k++) dot += Q[i*d+k] * K[j*d+k];
            S[i*N+j] = dot * scale;
        }

    // Softmax with optional causal mask
    for (int i = 0; i < N; i++) {
        int cols = causal ? (i+1) : N;
        float m = -FLT_MAX;
        for (int j = 0; j < cols; j++) m = fmaxf(m, S[i*N+j]);
        float s = 0.f;
        for (int j = 0; j < N; j++) {
            float v = (j < cols) ? expf(S[i*N+j] - m) : 0.f;
            S[i*N+j] = v; s += v;
        }
        for (int j = 0; j < N; j++) S[i*N+j] /= s;
    }

    // O = P @ V
    for (int i = 0; i < N; i++)
        for (int k = 0; k < d; k++) {
            float acc = 0.f;
            for (int j = 0; j < N; j++) acc += S[i*N+j] * V[j*d+k];
            O[i*d+k] = acc;
        }
    free(S);
}

// ─────────────────────────────────────────────────────────────────────────────
// Run naive attention and report memory stats
// ─────────────────────────────────────────────────────────────────────────────
static void runNaiveAttention(int N, int d, bool causal, bool verbose,
                               cublasHandle_t cublas) {
    size_t bytes_qkv = (size_t)N * d * sizeof(float);
    size_t bytes_s   = (size_t)N * N * sizeof(float);

    float *d_Q, *d_K, *d_V, *d_S, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, bytes_qkv));
    CUDA_CHECK(cudaMalloc(&d_K, bytes_qkv));
    CUDA_CHECK(cudaMalloc(&d_V, bytes_qkv));
    CUDA_CHECK(cudaMalloc(&d_S, bytes_s));
    CUDA_CHECK(cudaMalloc(&d_O, bytes_qkv));

    // Initialise Q, K, V with small random values
    {
        float* h = (float*)malloc(bytes_qkv);
        rand_fill(h, N*d, -0.1f, 0.1f);
        CUDA_CHECK(cudaMemcpy(d_Q, h, bytes_qkv, cudaMemcpyHostToDevice));
        rand_fill(h, N*d, -0.1f, 0.1f);
        CUDA_CHECK(cudaMemcpy(d_K, h, bytes_qkv, cudaMemcpyHostToDevice));
        rand_fill(h, N*d, -0.1f, 0.1f);
        CUDA_CHECK(cudaMemcpy(d_V, h, bytes_qkv, cudaMemcpyHostToDevice));
        free(h);
    }

    float alpha = 1.f / sqrtf((float)d);
    float beta  = 0.f;
    float one   = 1.f, zero = 0.f;
    const int BLK = 256;

    // Warm up
    cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        N, N, d, &alpha, d_K, d, d_Q, d, &beta, d_S, N);
    softmaxCausal<<<N, BLK, 2*BLK*sizeof(float)>>>(d_S, N, causal);
    cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
        d, N, N, &one, d_V, d, d_S, N, &zero, d_O, d);
    CUDA_CHECK(cudaDeviceSynchronize());

    const int REPS = (N <= 2048) ? 20 : 5;
    GpuTimer t;

    // Step 1+2: S = Q @ K^T / sqrt(d)  (scale folded into cuBLAS alpha)
    t.start();
    for (int r = 0; r < REPS; r++)
        cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            N, N, d, &alpha, d_K, d, d_Q, d, &beta, d_S, N);
    float ms_qk = t.stop_ms() / REPS;

    // Step 3: softmax
    t.start();
    for (int r = 0; r < REPS; r++)
        softmaxCausal<<<N, BLK, 2*BLK*sizeof(float)>>>(d_S, N, causal);
    float ms_sm = t.stop_ms() / REPS;

    // Step 4: O = P @ V
    t.start();
    for (int r = 0; r < REPS; r++)
        cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            d, N, N, &one, d_V, d, d_S, N, &zero, d_O, d);
    float ms_pv = t.stop_ms() / REPS;

    float ms_total = ms_qk + ms_sm + ms_pv;

    // Arithmetic intensity analysis
    // Q@K^T: 2*N*N*d FLOPs, reads 2*N*d, writes N*N
    double ai_qk = 2.0*N*N*d / (2.0*N*d*4 + (double)N*N*4);
    // softmax: ~3*N FLOPs/row, reads N*N twice, writes N*N
    double ai_sm = 3.0*N*N / (3.0*(double)N*N*4);
    // P@V: 2*N*N*d FLOPs, reads N*N + N*d, writes N*d
    double ai_pv = 2.0*N*N*d / ((double)N*N*4 + 2.0*N*d*4);

    if (verbose) {
        printf("  N=%-5d  d=%-3d  S_matrix=%.0f MB\n",
               N, d, (double)bytes_s/1e6);
        printf("  %-18s  %7.2f ms  AI=%5.1f  %s\n",
               "Q@K^T (step 1+2)", ms_qk, ai_qk,
               ai_qk > 20 ? "compute-bound" : "memory-bound");
        printf("  %-18s  %7.2f ms  AI=%5.2f  %s\n",
               "softmax (step 3)", ms_sm, ai_sm,
               ai_sm > 20 ? "compute-bound" : "memory-bound");
        printf("  %-18s  %7.2f ms  AI=%5.1f  %s\n",
               "P@V (step 4)", ms_pv, ai_pv,
               ai_pv > 20 ? "compute-bound" : "memory-bound");
        printf("  %-18s  %7.2f ms  total\n\n", "TOTAL", ms_total);
    } else {
        printf("  N=%-6d  S=%.0f MB  total=%.2f ms  "
               "(QK=%.2f  SM=%.2f  PV=%.2f)\n",
               N, (double)bytes_s/1e6, ms_total, ms_qk, ms_sm, ms_pv);
    }

    // Correctness check (only for small N)
    if (N <= 256 && verbose) {
        float* h_O_gpu = (float*)malloc(bytes_qkv);
        float* h_O_cpu = (float*)malloc(bytes_qkv);
        float* h_Q = (float*)malloc(bytes_qkv);
        float* h_K = (float*)malloc(bytes_qkv);
        float* h_V = (float*)malloc(bytes_qkv);
        CUDA_CHECK(cudaMemcpy(h_Q, d_Q, bytes_qkv, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_K, d_K, bytes_qkv, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_V, d_V, bytes_qkv, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_O_gpu, d_O, bytes_qkv, cudaMemcpyDeviceToHost));
        cpuAttention(h_Q, h_K, h_V, h_O_cpu, N, d, causal);
        printf("  Correctness vs CPU: max error = %.2e  %s\n\n",
               max_err(h_O_gpu, h_O_cpu, N*d),
               max_err(h_O_gpu, h_O_cpu, N*d) < 1e-3f ? "PASS" : "FAIL");
        free(h_O_gpu); free(h_O_cpu); free(h_Q); free(h_K); free(h_V);
    }

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_S); cudaFree(d_O);
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main() {
    print_device_info();
    srand(42);

    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));

    // Typical head dimension: 64 or 128
    const int D = 64;

    section("Naive attention: step-by-step breakdown (N=1024, d=64)");
    runNaiveAttention(1024, D, /*causal=*/true, /*verbose=*/true, cublas);

    section("Correctness check (N=128, causal)");
    runNaiveAttention(128, D, true, true, cublas);

    section("N^2 scaling: how the attention matrix grows with sequence length");
    printf("  %-8s %12s %12s %12s\n", "N", "S matrix", "Time (ms)", "vs N=512");
    printf("  %-8s %12s %12s %12s\n", "--------",
           "----------","----------","----------");

    float ms_base = 0;
    int sizes[] = {512, 1024, 2048, 4096, 8192};
    for (int i = 0; i < 5; i++) {
        int N = sizes[i];
        size_t s_mb = (size_t)N*N*4 / (1024*1024);  // in MB

        // Quick timing: run all three steps combined
        float *d_Q, *d_K, *d_V, *d_S, *d_O;
        size_t bqkv = (size_t)N*D*4, bs = (size_t)N*N*4;
        CUDA_CHECK(cudaMalloc(&d_Q, bqkv)); CUDA_CHECK(cudaMalloc(&d_K, bqkv));
        CUDA_CHECK(cudaMalloc(&d_V, bqkv)); CUDA_CHECK(cudaMalloc(&d_O, bqkv));
        CUDA_CHECK(cudaMalloc(&d_S, bs));
        CUDA_CHECK(cudaMemset(d_Q,1,bqkv)); CUDA_CHECK(cudaMemset(d_K,1,bqkv));
        CUDA_CHECK(cudaMemset(d_V,1,bqkv));

        float alpha = 1.f/sqrtf((float)D), beta=0.f, one=1.f, zero=0.f;
        const int BLK = 256;

        // Warm up
        cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            N, N, D, &alpha, d_K, D, d_Q, D, &beta, d_S, N);
        softmaxCausal<<<N, BLK, 2*BLK*sizeof(float)>>>(d_S, N, true);
        cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            D, N, N, &one, d_V, D, d_S, N, &zero, d_O, D);
        CUDA_CHECK(cudaDeviceSynchronize());

        int reps = (N <= 2048) ? 10 : 3;
        GpuTimer t; t.start();
        for (int r = 0; r < reps; r++) {
            cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                N, N, D, &alpha, d_K, D, d_Q, D, &beta, d_S, N);
            softmaxCausal<<<N, BLK, 2*BLK*sizeof(float)>>>(d_S, N, true);
            cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                D, N, N, &one, d_V, D, d_S, N, &zero, d_O, D);
        }
        float ms = t.stop_ms() / reps;

        if (i == 0) ms_base = ms;
        printf("  %-8d %10zu MB %12.2f %12.2fx\n",
               N, s_mb, ms, ms / ms_base);

        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
        cudaFree(d_S); cudaFree(d_O);
    }

    printf("\n  Observation: time scales as ~O(N^2) — doubling N quadruples time.\n");
    printf("  The S matrix at N=8192 is 256 MB per head.\n");
    printf("  With 32 heads and a typical 4K batch: OOM on H100 80GB.\n");
    printf("  FlashAttention solves this: next program (03_flash_attention.cu).\n");

    CUBLAS_CHECK(cublasDestroy(cublas));
    return 0;
}
