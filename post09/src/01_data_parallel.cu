// 01_data_parallel.cu  --  Post 9: Data and Tensor Parallelism
//
// Data parallelism (DDP) is the simplest and most common way to scale training:
//   - Every GPU holds a FULL copy of the model
//   - The global batch is split into per-GPU shards
//   - Each GPU computes gradients on its own shard
//   - Gradients are AllReduced (summed, then averaged) across all GPUs
//   - Every GPU applies the identical averaged update -> models stay in sync
//
// The defining property: DDP on P GPUs with batch B/P each must produce the
// SAME gradient as a single GPU processing the full batch B. This program
// verifies exactly that equivalence.
//
// We simulate P "virtual GPUs" with separate buffers on one physical GPU.
// Each computes gradients on its batch shard; we AllReduce (sum) them and
// compare to the single-GPU full-batch gradient. They must match.
//
// Layer used: a single linear layer y = x @ W^T, loss = sum(y), so the
// gradient dL/dW = sum over batch of (1-vector outer x). Simple enough to
// verify exactly, structurally identical to a real layer's gradient.

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include "../include/utils.cuh"

// dL/dW for loss = sum(y), y = x @ W^T  is  dL/dW[o,i] = sum_b x[b,i]
// (every output gets gradient 1, so dW = ones[Dout,B] @ x[B,Din] = colsum(x)
//  broadcast across output rows). We compute it as a matmul: dW = G^T @ X
// where G = ones[B x Dout]. This mirrors the real backward dW = dY^T @ X.
__global__ void fillOnes(float* p, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = 1.0f;
}

// Scale a buffer in place (used to average: divide summed gradient by P)
__global__ void scaleInPlace(float* p, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] *= s;
}

// Element-wise accumulate: dst += src
__global__ void accumulate(float* dst, const float* src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
}

int main(int argc, char** argv) {
    print_device_info();
    srand(42);

    const int P     = (argc > 1) ? atoi(argv[1]) : 4;   // simulated GPUs
    const int Bglob = (argc > 2) ? atoi(argv[2]) : 256; // global batch
    const int Din   = 512;
    const int Dout  = 256;
    const int BLK   = 256;

    const int Bshard = Bglob / P;   // per-GPU batch

    printf("Data-parallel setup: P=%d GPUs, global batch=%d (%d per GPU)\n",
           P, Bglob, Bshard);
    printf("Layer: linear [%d -> %d]\n\n", Din, Dout);

    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));
    float one = 1.f, zero = 0.f;

    // ── Generate the full global batch on the host ───────────────────────────
    float* h_X = (float*)malloc((size_t)Bglob * Din * sizeof(float));
    rand_fill(h_X, Bglob * Din, -1.f, 1.f);

    // ── Reference: single GPU processes the FULL batch ───────────────────────
    section("Reference: single GPU, full batch");
    float* d_X_full;   // [Bglob x Din]
    float* d_G_full;   // [Bglob x Dout] upstream grad (all ones)
    float* d_dW_ref;   // [Dout x Din] reference gradient
    CUDA_CHECK(cudaMalloc(&d_X_full,  (size_t)Bglob * Din  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_G_full,  (size_t)Bglob * Dout * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dW_ref,  (size_t)Dout  * Din  * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_X_full, h_X, (size_t)Bglob * Din * sizeof(float),
                          cudaMemcpyHostToDevice));
    fillOnes<<<(Bglob*Dout+BLK-1)/BLK, BLK>>>(d_G_full, Bglob*Dout);

    // dW = G^T @ X : [Dout x B] @ [B x Din] = [Dout x Din]
    // cuBLAS col-major trick: X^T[Din x B] @ G[B x Dout] -> dW^T[Din x Dout]
    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
        Din, Dout, Bglob, &one, d_X_full, Din, d_G_full, Dout, &zero, d_dW_ref, Din));
    // Average over the global batch (standard mean loss)
    scaleInPlace<<<(Dout*Din+BLK-1)/BLK, BLK>>>(d_dW_ref, 1.f/Bglob, Dout*Din);
    CUDA_CHECK(cudaDeviceSynchronize());

    float* h_dW_ref = (float*)malloc((size_t)Dout * Din * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_dW_ref, d_dW_ref, (size_t)Dout*Din*sizeof(float),
                          cudaMemcpyDeviceToHost));
    printf("  Full batch B=%d -> single gradient dW [%d x %d]\n", Bglob, Dout, Din);

    // ── Data-parallel: P GPUs, each on a batch shard, then AllReduce ─────────
    section("Data-parallel: P shards + gradient AllReduce");

    float** d_X    = (float**)malloc(P * sizeof(float*));  // per-GPU input shard
    float** d_G    = (float**)malloc(P * sizeof(float*));  // per-GPU upstream grad
    float** d_dW   = (float**)malloc(P * sizeof(float*));  // per-GPU local gradient

    for (int r = 0; r < P; r++) {
        CUDA_CHECK(cudaMalloc(&d_X[r],  (size_t)Bshard * Din  * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_G[r],  (size_t)Bshard * Dout * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dW[r], (size_t)Dout   * Din  * sizeof(float)));
        // Copy this GPU's batch shard from the full batch
        CUDA_CHECK(cudaMemcpy(d_X[r], h_X + (size_t)r * Bshard * Din,
                              (size_t)Bshard * Din * sizeof(float),
                              cudaMemcpyHostToDevice));
        fillOnes<<<(Bshard*Dout+BLK-1)/BLK, BLK>>>(d_G[r], Bshard*Dout);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Each GPU computes its LOCAL gradient on its shard
    for (int r = 0; r < P; r++) {
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
            Din, Dout, Bshard, &one, d_X[r], Din, d_G[r], Dout, &zero, d_dW[r], Din));
        // Each shard scales by 1/Bglob so the SUM equals the mean over the full batch
        scaleInPlace<<<(Dout*Din+BLK-1)/BLK, BLK>>>(d_dW[r], 1.f/Bglob, Dout*Din);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // AllReduce (sum) the local gradients into GPU 0's buffer
    // (In real DDP, NCCL ring-AllReduce does this; here we sum directly.)
    for (int r = 1; r < P; r++)
        accumulate<<<(Dout*Din+BLK-1)/BLK, BLK>>>(d_dW[0], d_dW[r], Dout*Din);
    CUDA_CHECK(cudaDeviceSynchronize());

    float* h_dW_dp = (float*)malloc((size_t)Dout * Din * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_dW_dp, d_dW[0], (size_t)Dout*Din*sizeof(float),
                          cudaMemcpyDeviceToHost));

    // ── Verify the gradients match ────────────────────────────────────────────
    float err = max_abs_diff(h_dW_ref, h_dW_dp, Dout * Din);
    printf("  Each GPU: gradient on %d-sample shard, scaled by 1/%d\n", Bshard, Bglob);
    printf("  AllReduce (sum) of %d local gradients\n", P);
    printf("  Max diff vs single-GPU full-batch gradient: %.2e  %s\n", err,
           err < 1e-4f ? "PASS -- DDP is mathematically equivalent" : "FAIL");

    // ── Memory cost ───────────────────────────────────────────────────────────
    section("DDP memory cost (the key limitation)");
    double model_mb = (double)Dout * Din * sizeof(float) / 1e6;
    printf("  Every GPU holds a FULL model replica: %.2f MB (this layer)\n", model_mb);
    printf("  DDP does NOT reduce per-GPU memory -- it only splits the batch.\n");
    printf("  A 7B model needs ~14 GB on EVERY GPU just for weights,\n");
    printf("  plus optimizer state (~56 GB for Adam in FP32) on EVERY GPU.\n");
    printf("  This is what ZeRO (program 02) fixes by sharding that state.\n\n");

    printf("  Communication per step: AllReduce of the full gradient.\n");
    printf("    Gradient size = model size. For 7B BF16: 14 GB AllReduced/step.\n");
    printf("    On NVLink (900 GB/s): ~30 ms. Hidden if compute step > 30 ms.\n");

    // ── How PyTorch does it ───────────────────────────────────────────────────
    section("How PyTorch DDP maps to this");
    printf("  model = DistributedDataParallel(model)\n");
    printf("  - Replicates the model to every GPU (the full copy above)\n");
    printf("  - Splits the DataLoader via DistributedSampler (batch shards)\n");
    printf("  - Registers backward hooks that fire ncclAllReduce on gradients\n");
    printf("  - Overlaps gradient AllReduce with backward compute (bucketing)\n");
    printf("  The averaged gradient is identical to single-GPU full-batch.\n");

    // Cleanup
    cublasDestroy(cublas);
    cudaFree(d_X_full); cudaFree(d_G_full); cudaFree(d_dW_ref);
    for (int r = 0; r < P; r++) { cudaFree(d_X[r]); cudaFree(d_G[r]); cudaFree(d_dW[r]); }
    free(d_X); free(d_G); free(d_dW);
    free(h_X); free(h_dW_ref); free(h_dW_dp);
    return 0;
}
