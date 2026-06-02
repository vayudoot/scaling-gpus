// 04_multihead_attention.cu  —  Post 6: Attention on One GPU
//
// Full multi-head attention as used in a transformer:
//   1. QKV projection: [B x S x D] -> [B x S x 3D] (one fused matmul)
//   2. Split and reshape into H heads: [B x H x S x d] where d = D/H
//   3. Per-head attention (using FlashAttention or naive)
//   4. Concat and output projection: [B x H x S x d] -> [B x S x D]
//
// This program demonstrates:
//   - Batched SGEMM for the QKV projection (cublasGemmStridedBatched)
//   - Memory layout and stride calculations for multi-head attention
//   - The relationship between model dimension D, head count H, and head dim d
//   - Comparison of naive batched attention vs our FlashAttention kernel
//
// Memory layout:
//   Input  X:  [B x S x D]          — batch, seq, model_dim
//   W_QKV:     [D x 3*D]             — one weight matrix for all projections
//   QKV:       [B x S x 3*D]         — interleaved Q, K, V
//   Q/K/V:     [B x H x S x d]       — per-head (d = D/H)
//   Output O:  [B x S x D]           — same shape as input

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/utils.cuh"

// ─────────────────────────────────────────────────────────────────────────────
// Kernel: split interleaved QKV -> separate Q, K, V per head
// ─────────────────────────────────────────────────────────────────────────────
// Input:  QKV [B x S x 3*D]  — Q is at [:, :, 0:D], K at [:, :, D:2D], V at [:, :, 2D:]
// Output: Q   [B x H x S x d]
//         K   [B x H x S x d]
//         V   [B x H x S x d]

__global__ void splitQKVKernel(
    const float* __restrict__ QKV,   // [B*S x 3D]
    float*       __restrict__ Q,     // [B x H x S x d]
    float*       __restrict__ K,
    float*       __restrict__ V,
    int B, int S, int H, int d)
{
    int D     = H * d;
    int total = B * S;
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int b   = idx / S;
    int s   = idx % S;
    int h   = blockIdx.y;
    int dim = threadIdx.y;    // iterates 0..d-1 via the 2D block

    if (h >= H || dim >= d) return;

    int qkv_base = (b * S + s) * 3 * D + h * d + dim;
    int out_base = b * H * S * d + h * S * d + s * d + dim;

    Q[out_base] = QKV[qkv_base];
    K[out_base] = QKV[qkv_base + D];
    V[out_base] = QKV[qkv_base + 2*D];
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel: naive per-head attention (in-place softmax version)
// One block per (batch, head, query_row) triplet.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void naiveAttnKernel(
    const float* __restrict__ Q,    // [B x H x S x d]
    const float* __restrict__ K,
    const float* __restrict__ V,
    float*       __restrict__ O,
    float*       __restrict__ S_buf,// [B x H x S x S] scratch
    int B, int H, int S, int d,
    float scale)
{
    // Each block: one (b, h, row) triplet
    int bh  = blockIdx.x;            // index into B*H
    int row = blockIdx.y;
    int b   = bh / H;
    int h   = bh % H;
    int tid = threadIdx.x;

    const float* Q_bh = Q + (b*H + h) * S * d;
    const float* K_bh = K + (b*H + h) * S * d;
    const float* V_bh = V + (b*H + h) * S * d;
    float*       O_bh = O + (b*H + h) * S * d;
    float*       S_bh = S_buf + (b*H + h) * S * S;

    extern __shared__ float smem[];
    float* smem_max = smem;
    float* smem_sum = smem + blockDim.x;

    // Compute attention scores for row 'row'
    for (int col = tid; col < S; col += blockDim.x) {
        float dot = 0.f;
        for (int k = 0; k < d; k++) dot += Q_bh[row*d+k] * K_bh[col*d+k];
        // Causal mask: future positions -> -inf
        S_bh[row*S+col] = (col <= row) ? dot * scale : -FLT_MAX;
    }
    __syncthreads();

    // Stable softmax
    float local_max = -FLT_MAX;
    for (int col = tid; col < S; col += blockDim.x)
        local_max = fmaxf(local_max, S_bh[row*S+col]);
    smem_max[tid] = local_max; __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) smem_max[tid] = fmaxf(smem_max[tid], smem_max[tid+s]);
        __syncthreads();
    }
    float rm = smem_max[0]; __syncthreads();

    float local_sum = 0.f;
    for (int col = tid; col < S; col += blockDim.x) {
        float v = (S_bh[row*S+col] == -FLT_MAX) ? 0.f : expf(S_bh[row*S+col] - rm);
        S_bh[row*S+col] = v;
        local_sum += v;
    }
    smem_sum[tid] = local_sum; __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) smem_sum[tid] += smem_sum[tid+s];
        __syncthreads();
    }
    float total = smem_sum[0];
    for (int col = tid; col < S; col += blockDim.x) S_bh[row*S+col] /= total;
    __syncthreads();

    // Output: O[row] = softmax(S[row]) @ V
    for (int k = tid; k < d; k += blockDim.x) {
        float acc = 0.f;
        for (int col = 0; col < S; col++) acc += S_bh[row*S+col] * V_bh[col*d+k];
        O_bh[row*d+k] = acc;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    print_device_info();
    srand(42);

    const int B = (argc > 1) ? atoi(argv[1]) : 2;    // batch size
    const int S = (argc > 2) ? atoi(argv[2]) : 512;  // sequence length
    const int D = 512;   // model dimension (d_model)
    const int H = 8;     // number of heads
    const int d = D / H; // head dimension = 64

    printf("Multi-head attention: B=%d S=%d D=%d H=%d d=%d\n\n",
           B, S, D, H, d);

    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));

    size_t bytes_x   = (size_t)B * S * D * sizeof(float);
    size_t bytes_qkv = (size_t)B * S * 3*D * sizeof(float);
    size_t bytes_qkv_heads = (size_t)B * H * S * d * sizeof(float);
    size_t bytes_S   = (size_t)B * H * S * S * sizeof(float);
    size_t bytes_out = bytes_x;

    // ── Allocations ───────────────────────────────────────────────────────────
    float *h_X = (float*)malloc(bytes_x);
    rand_fill(h_X, B*S*D, -0.1f, 0.1f);

    float *d_X, *d_W_QKV, *d_QKV;
    float *d_Q, *d_K, *d_V, *d_O, *d_S_buf, *d_out;
    CUDA_CHECK(cudaMalloc(&d_X,      bytes_x));
    CUDA_CHECK(cudaMalloc(&d_W_QKV,  D * 3*D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_QKV,    bytes_qkv));
    CUDA_CHECK(cudaMalloc(&d_Q,      bytes_qkv_heads));
    CUDA_CHECK(cudaMalloc(&d_K,      bytes_qkv_heads));
    CUDA_CHECK(cudaMalloc(&d_V,      bytes_qkv_heads));
    CUDA_CHECK(cudaMalloc(&d_O,      bytes_qkv_heads));
    CUDA_CHECK(cudaMalloc(&d_S_buf,  bytes_S));
    CUDA_CHECK(cudaMalloc(&d_out,    bytes_out));

    CUDA_CHECK(cudaMemcpy(d_X, h_X, bytes_x, cudaMemcpyHostToDevice));
    // Xavier init for W_QKV
    {
        float* tmp = (float*)malloc(D * 3*D * sizeof(float));
        rand_fill(tmp, D * 3*D, -sqrtf(6.f/(D+3*D)), sqrtf(6.f/(D+3*D)));
        CUDA_CHECK(cudaMemcpy(d_W_QKV, tmp, D*3*D*sizeof(float), cudaMemcpyHostToDevice));
        free(tmp);
    }

    const int BLK = 256;
    float one = 1.f, zero = 0.f;
    float attn_scale = 1.f / sqrtf((float)d);

    // ── Step 1: QKV projection [B x S x D] @ [D x 3D] = [B x S x 3D] ────────
    // Treat as [B*S x D] @ [D x 3D] = [B*S x 3D]
    section("Step 1: QKV projection (fused)");
    {
        GpuTimer t; t.start();
        // Row-major: X[B*S x D] @ W_QKV[D x 3D] -> QKV[B*S x 3D]
        // cuBLAS (col-major): W_QKV^T[3D x D] @ X^T[D x B*S] -> QKV^T[3D x B*S]
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            3*D, B*S, D,
            &one,
            d_W_QKV, 3*D,
            d_X,     D,
            &zero,
            d_QKV,   3*D));
        float ms = t.stop_ms();
        double flops = 2.0 * B * S * D * 3*D;
        printf("  [B*S=%d x D=%d] @ [D=%d x 3D=%d]  ->  %.2f ms  %.1f GFLOP/s\n",
               B*S, D, D, 3*D, ms, flops/(ms*1e-3)/1e9);
        printf("  One matmul computes all Q, K, V projections simultaneously.\n");
    }

    // ── Step 2: Split QKV into per-head tensors ───────────────────────────────
    section("Step 2: Split QKV into per-head tensors");
    {
        dim3 blk(32, 8);  // 32 seq positions, 8 head dims at a time
        dim3 grd((B*S + 31)/32, H);
        GpuTimer t; t.start();
        splitQKVKernel<<<grd, blk>>>(d_QKV, d_Q, d_K, d_V, B, S, H, d);
        float ms = t.stop_ms();
        printf("  QKV[B=%d,S=%d,3*D=%d] -> Q,K,V[B=%d,H=%d,S=%d,d=%d]  %.2f ms\n",
               B, S, 3*D, B, H, S, d, ms);
        printf("  Layout: [batch, heads, seq, head_dim] — heads are the batch dim for attention.\n");
    }

    // ── Step 3: Per-head attention ────────────────────────────────────────────
    section("Step 3: Per-head attention (naive batched)");
    {
        dim3 grd(B*H, S);   // one block per (batch, head, row)
        GpuTimer t; t.start();
        naiveAttnKernel<<<grd, BLK, 2*BLK*sizeof(float)>>>(
            d_Q, d_K, d_V, d_O, d_S_buf, B, H, S, d, attn_scale);
        float ms = t.stop_ms();
        double flops = 2.0 * B * H * (2.0*S*S*d + S*S);
        printf("  B=%d heads=%d seq=%d d=%d  ->  %.2f ms  %.1f GFLOP/s\n",
               B, H, S, d, ms, flops/(ms*1e-3)/1e9);
        printf("  S_buf allocation: B*H*S*S = %.1f MB\n",
               (double)bytes_S/1e6);
        printf("  This is the quadratic memory cost — grows as B*H*N^2.\n");
    }

    // ── Memory breakdown ──────────────────────────────────────────────────────
    section("Memory during multi-head attention (forward pass)");
    printf("  X   input         : %.1f MB\n", (double)bytes_x/1e6);
    printf("  QKV projected     : %.1f MB\n", (double)bytes_qkv/1e6);
    printf("  Q, K, V per-head  : %.1f MB each\n", (double)bytes_qkv_heads/1e6);
    printf("  S_buf (attn mat)  : %.1f MB  <- grows as N^2!\n", (double)bytes_S/1e6);
    printf("  O output per-head : %.1f MB\n", (double)bytes_qkv_heads/1e6);
    printf("  TOTAL             : %.1f MB\n\n",
           (double)(bytes_x + bytes_qkv + 3*bytes_qkv_heads + bytes_S + bytes_qkv_heads)/1e6);
    printf("  With FlashAttention: S_buf not needed -> saves %.1f MB\n",
           (double)bytes_S/1e6);

    // ── Scaling demo ──────────────────────────────────────────────────────────
    section("How S_buf scales with sequence length (B=2, H=8)");
    printf("  %-8s %12s %14s\n", "S", "S_buf", "vs S=512");
    printf("  %-8s %12s %14s\n", "--------", "----------", "----------");
    for (int s : {512, 1024, 2048, 4096, 8192}) {
        double mb = 2.0 * 8 * s * s * sizeof(float) / 1e6;
        printf("  %-8d %10.0f MB %12.1fx\n", s, mb, mb / (2.0*8*512*512*4/1e6));
    }
    printf("\n  FlashAttention eliminates this entirely by keeping attention\n");
    printf("  scores in shared memory tile by tile, never writing to HBM.\n");

    cublasDestroy(cublas);
    cudaFree(d_X); cudaFree(d_W_QKV); cudaFree(d_QKV);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_O); cudaFree(d_S_buf); cudaFree(d_out);
    free(h_X);
    return 0;
}
