// 03_tensor_parallel.cu  --  Post 9: Data and Tensor Parallelism
//
// Tensor parallelism splits a single layer's weight matrix across GPUs so a
// layer too large for one GPU can still run. This is the Megatron-LM approach,
// used for every model whose individual layers exceed one GPU's memory.
//
// There are two ways to split a linear layer  Y = X @ W^T:
//
//   COLUMN-PARALLEL: split W by its output dimension (columns of W^T).
//     GPU r holds W_r = columns [r*d : (r+1)*d] of the output.
//     Each GPU computes Y_r = X @ W_r^T independently -- NO communication.
//     Result: Y is split across GPUs by column. Perfect for the FIRST linear
//     of an MLP because the split output feeds directly into the next layer.
//
//   ROW-PARALLEL: split W by its input dimension (rows of W^T / columns of W).
//     GPU r holds W_r = rows [r*d : (r+1)*d] of the input, and the matching
//     slice X_r of the input. Each computes a PARTIAL sum Y_r = X_r @ W_r^T.
//     The partial sums must be ADDED across GPUs -> ONE AllReduce.
//     Perfect for the SECOND linear of an MLP because it consumes a
//     column-split input and produces a complete output.
//
// The Megatron MLP trick:  Y = Dropout( GeLU(X @ A) @ B )
//   - A is column-parallel  (splits the hidden dimension, no comm)
//   - B is row-parallel     (consumes the split hidden dim, one AllReduce)
//   The entire MLP block needs only ONE AllReduce in the forward pass.
//
// This program implements both splits, composes them into the Megatron MLP,
// and verifies the tensor-parallel result equals a single-GPU reference.

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/utils.cuh"

// GeLU activation (matches PyTorch's default tanh approximation)
__global__ void geluKernel(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        float c = 0.7978845608f * (v + 0.044715f * v * v * v);
        x[i] = 0.5f * v * (1.f + tanhf(c));
    }
}

// dst += src
__global__ void accumulate(float* dst, const float* src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
}

// ─────────────────────────────────────────────────────────────────────────────
// CPU reference: single-GPU MLP  Y = GeLU(X @ A) @ B
//   X [B x H],  A [H x 4H],  B [4H x H]  ->  Y [B x H]
// ─────────────────────────────────────────────────────────────────────────────
static void cpuMLP(const float* X, const float* A, const float* B,
                   float* Y, int Bsz, int H, int Hff) {
    float* hidden = (float*)malloc((size_t)Bsz * Hff * sizeof(float));
    // hidden = GeLU(X @ A)
    for (int b = 0; b < Bsz; b++)
        for (int j = 0; j < Hff; j++) {
            float s = 0.f;
            for (int k = 0; k < H; k++) s += X[b*H+k] * A[k*Hff+j];
            float c = 0.7978845608f * (s + 0.044715f*s*s*s);
            hidden[b*Hff+j] = 0.5f * s * (1.f + tanhf(c));
        }
    // Y = hidden @ B
    for (int b = 0; b < Bsz; b++)
        for (int j = 0; j < H; j++) {
            float s = 0.f;
            for (int k = 0; k < Hff; k++) s += hidden[b*Hff+k] * B[k*H+j];
            Y[b*H+j] = s;
        }
    free(hidden);
}

int main(int argc, char** argv) {
    print_device_info();
    srand(42);

    const int P    = (argc > 1) ? atoi(argv[1]) : 4;   // tensor-parallel GPUs
    const int Bsz  = 64;       // batch (rows)
    const int H    = 512;      // model hidden dim
    const int Hff  = 4 * H;    // MLP intermediate dim (must divide by P)
    const int BLK  = 256;

    printf("Tensor-parallel MLP: P=%d GPUs\n", P);
    printf("Y = GeLU(X @ A) @ B   X[%d x %d]  A[%d x %d]  B[%d x %d]\n\n",
           Bsz, H, H, Hff, Hff, H);
    printf("A is column-parallel (split %d -> %d per GPU, no comm)\n", Hff, Hff/P);
    printf("B is row-parallel    (split %d -> %d per GPU, one AllReduce)\n\n", Hff, Hff/P);

    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));
    float one = 1.f, zero = 0.f;

    // ── Host data ─────────────────────────────────────────────────────────────
    float* h_X = (float*)malloc((size_t)Bsz * H   * sizeof(float));
    float* h_A = (float*)malloc((size_t)H   * Hff * sizeof(float));
    float* h_B = (float*)malloc((size_t)Hff * H   * sizeof(float));
    rand_fill(h_X, Bsz*H,   -0.5f, 0.5f);
    rand_fill(h_A, H*Hff,   -0.1f, 0.1f);
    rand_fill(h_B, Hff*H,   -0.1f, 0.1f);

    // ── Reference ─────────────────────────────────────────────────────────────
    section("Reference: single-GPU MLP (CPU)");
    float* h_Y_ref = (float*)malloc((size_t)Bsz * H * sizeof(float));
    cpuMLP(h_X, h_A, h_B, h_Y_ref, Bsz, H, Hff);
    printf("  Computed full MLP on CPU as ground truth.\n");

    // ── Tensor-parallel forward ───────────────────────────────────────────────
    section("Tensor-parallel forward across P virtual GPUs");
    const int Hff_shard = Hff / P;   // each GPU's slice of the intermediate dim

    // X is replicated on every GPU (it's the layer input)
    float* d_X;
    CUDA_CHECK(cudaMalloc(&d_X, (size_t)Bsz * H * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_X, h_X, (size_t)Bsz*H*sizeof(float), cudaMemcpyHostToDevice));

    // Per-GPU buffers
    float** d_A_shard = (float**)malloc(P * sizeof(float*)); // [H x Hff_shard]
    float** d_B_shard = (float**)malloc(P * sizeof(float*)); // [Hff_shard x H]
    float** d_hidden  = (float**)malloc(P * sizeof(float*)); // [Bsz x Hff_shard]
    float** d_Y_part  = (float**)malloc(P * sizeof(float*)); // [Bsz x H] partial

    for (int r = 0; r < P; r++) {
        CUDA_CHECK(cudaMalloc(&d_A_shard[r], (size_t)H        * Hff_shard * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B_shard[r], (size_t)Hff_shard * H        * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_hidden[r],  (size_t)Bsz      * Hff_shard * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_Y_part[r],  (size_t)Bsz      * H         * sizeof(float)));

        // COLUMN-PARALLEL split of A: GPU r gets columns [r*Hff_shard : ...]
        // A is [H x Hff] row-major; column slice = A[:, r*Hff_shard : (r+1)*Hff_shard]
        float* h_A_shard = (float*)malloc((size_t)H * Hff_shard * sizeof(float));
        for (int k = 0; k < H; k++)
            for (int j = 0; j < Hff_shard; j++)
                h_A_shard[k*Hff_shard + j] = h_A[k*Hff + r*Hff_shard + j];
        CUDA_CHECK(cudaMemcpy(d_A_shard[r], h_A_shard,
                              (size_t)H*Hff_shard*sizeof(float), cudaMemcpyHostToDevice));
        free(h_A_shard);

        // ROW-PARALLEL split of B: GPU r gets rows [r*Hff_shard : ...]
        // B is [Hff x H] row-major; row slice = B[r*Hff_shard : (r+1)*Hff_shard, :]
        CUDA_CHECK(cudaMemcpy(d_B_shard[r],
                              h_B + (size_t)r * Hff_shard * H,
                              (size_t)Hff_shard * H * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    // Step 1: each GPU computes hidden_r = GeLU(X @ A_r)  -- NO communication
    //   X[Bsz x H] @ A_r[H x Hff_shard] = hidden_r[Bsz x Hff_shard]
    for (int r = 0; r < P; r++) {
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            Hff_shard, Bsz, H, &one,
            d_A_shard[r], Hff_shard, d_X, H, &zero, d_hidden[r], Hff_shard));
        geluKernel<<<(Bsz*Hff_shard+BLK-1)/BLK, BLK>>>(d_hidden[r], Bsz*Hff_shard);
    }

    // Step 2: each GPU computes Y_r = hidden_r @ B_r  -- PARTIAL sums
    //   hidden_r[Bsz x Hff_shard] @ B_r[Hff_shard x H] = Y_r[Bsz x H]
    for (int r = 0; r < P; r++) {
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            H, Bsz, Hff_shard, &one,
            d_B_shard[r], H, d_hidden[r], Hff_shard, &zero, d_Y_part[r], H));
    }

    // Step 3: AllReduce the partial sums -> the complete output.
    // (Real TP uses ncclAllReduce; here we sum the partials directly.)
    for (int r = 1; r < P; r++)
        accumulate<<<(Bsz*H+BLK-1)/BLK, BLK>>>(d_Y_part[0], d_Y_part[r], Bsz*H);
    CUDA_CHECK(cudaDeviceSynchronize());

    float* h_Y_tp = (float*)malloc((size_t)Bsz * H * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_Y_tp, d_Y_part[0], (size_t)Bsz*H*sizeof(float),
                          cudaMemcpyDeviceToHost));

    // ── Verify ────────────────────────────────────────────────────────────────
    float err = max_abs_diff(h_Y_ref, h_Y_tp, Bsz * H);
    printf("  Step 1: hidden_r = GeLU(X @ A_r)  [column-parallel, 0 comm]\n");
    printf("  Step 2: Y_r = hidden_r @ B_r      [row-parallel, partial sums]\n");
    printf("  Step 3: AllReduce(Y_0..Y_{P-1})   [1 collective]\n\n");
    printf("  Max diff vs single-GPU reference: %.2e  %s\n", err,
           err < 1e-3f ? "PASS -- TP is mathematically exact" : "FAIL");

    // ── Communication and memory analysis ─────────────────────────────────────
    section("Tensor-parallel cost analysis");
    printf("  Communication: exactly ONE AllReduce per MLP block (forward).\n");
    printf("    Size = output activation = Bsz x H = %.2f MB\n",
           (double)Bsz * H * sizeof(float) / 1e6);
    printf("    Backward adds one more AllReduce. Two per block per step.\n\n");

    printf("  Memory: each GPU stores 1/P of A and 1/P of B:\n");
    printf("    A shard: %.2f MB (vs %.2f MB full)\n",
           (double)H*Hff_shard*sizeof(float)/1e6, (double)H*Hff*sizeof(float)/1e6);
    printf("    B shard: %.2f MB (vs %.2f MB full)\n",
           (double)Hff_shard*H*sizeof(float)/1e6, (double)Hff*H*sizeof(float)/1e6);
    printf("    -> A layer too big for one GPU now fits across P GPUs.\n\n");

    printf("  WHY this split order matters:\n");
    printf("    Column-parallel A produces a column-split hidden state with NO comm.\n");
    printf("    Row-parallel B consumes that split directly -- the only comm is\n");
    printf("    the final AllReduce. Reversing the order would need an extra\n");
    printf("    AllGather between the layers. Megatron's ordering is optimal.\n\n");

    printf("  CRITICAL constraint: TP communicates inside every layer, so it\n");
    printf("  needs the fastest links (NVLink). TP is kept WITHIN a node;\n");
    printf("  crossing nodes over InfiniBand would make the per-layer AllReduce\n");
    printf("  dominate. This is why TP degree usually equals GPUs-per-node (8).\n");

    // Cleanup
    cublasDestroy(cublas);
    cudaFree(d_X);
    for (int r = 0; r < P; r++) {
        cudaFree(d_A_shard[r]); cudaFree(d_B_shard[r]);
        cudaFree(d_hidden[r]); cudaFree(d_Y_part[r]);
    }
    free(d_A_shard); free(d_B_shard); free(d_hidden); free(d_Y_part);
    free(h_X); free(h_A); free(h_B); free(h_Y_ref); free(h_Y_tp);
    return 0;
}
