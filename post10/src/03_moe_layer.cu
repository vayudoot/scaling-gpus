// 03_moe_layer.cu  --  Post 10: Pipeline Parallelism and MoE
//
// A complete Mixture-of-Experts layer on one GPU, top-1 routing:
//
//   1. ROUTER:   logits = X @ Wr  ->  expert_id[t] = argmax, gate[t] = softmax prob
//   2. GROUP:    permute tokens so each expert's tokens are contiguous
//                (this permutation IS the AllToAll when experts live on
//                 different GPUs -- here it is a local scatter)
//   3. EXPERTS:  one FFN GEMM pair per expert, on its contiguous slice
//   4. UNGROUP:  scatter results back to original token order, scale by gate
//
// Verified against a straightforward per-token reference on the CPU.
//
// Then the two production headaches, measured on this very batch:
//   - LOAD IMBALANCE: the tokens-per-expert histogram (routing is learned,
//     nothing forces it uniform)
//   - CAPACITY FACTOR: cap tokens/expert at cf * N/E and count what drops

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/utils.cuh"

// ─────────────────────────────────────────────────────────────────────────────
// Router: per-token argmax over E logits + softmax gate of the winner
// ─────────────────────────────────────────────────────────────────────────────
__global__ void routerArgmax(const float* __restrict__ logits,   // [N x E]
                             int*  __restrict__ expert_id,       // [N]
                             float* __restrict__ gate,           // [N]
                             int N, int E) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= N) return;
    const float* row = logits + (size_t)t * E;
    int   best = 0;
    float mx   = row[0];
    for (int e = 1; e < E; e++)
        if (row[e] > mx) { mx = row[e]; best = e; }
    float denom = 0.f;
    for (int e = 0; e < E; e++) denom += expf(row[e] - mx);
    expert_id[t] = best;
    gate[t]      = 1.f / denom;          // = exp(mx-mx)/sum = softmax prob of winner
}

// count tokens per expert
__global__ void histKernel(const int* __restrict__ expert_id,
                           int* __restrict__ counts, int N) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < N) atomicAdd(&counts[expert_id[t]], 1);
}

// scatter tokens into expert-contiguous order; records each token's slot
__global__ void scatterKernel(const float* __restrict__ X,       // [N x D]
                              float* __restrict__ Xp,            // [N x D] permuted
                              const int* __restrict__ expert_id, // [N]
                              const int* __restrict__ base,      // [E] exclusive scan
                              int*  __restrict__ cursor,         // [E] running offset
                              int*  __restrict__ slot,           // [N] token -> row in Xp
                              int N, int D) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= N) return;
    int e   = expert_id[t];
    int pos = base[e] + atomicAdd(&cursor[e], 1);
    slot[t] = pos;
    const float* src = X  + (size_t)t   * D;
    float*       dst = Xp + (size_t)pos * D;
    for (int j = 0; j < D; j++) dst[j] = src[j];
}

// gather results back to token order, scaling by the gate value
__global__ void gatherKernel(const float* __restrict__ Yp,   // [N x D] permuted
                             float* __restrict__ Y,          // [N x D]
                             const int* __restrict__ slot,   // [N]
                             const float* __restrict__ gate, // [N]
                             int N, int D) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= N) return;
    const float* src = Yp + (size_t)slot[t] * D;
    float*       dst = Y  + (size_t)t       * D;
    float g = gate[t];
    for (int j = 0; j < D; j++) dst[j] = g * src[j];
}

__global__ void reluKernel(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && x[i] < 0.f) x[i] = 0.f;
}

int main(int argc, char** argv) {
    print_device_info();
    srand(42);

    const int N = 4096;        // tokens
    const int D = 512;         // model dim
    const int H = 1024;        // expert hidden dim
    const int E = 8;           // experts
    const int BLK = 256;
    const int N_VERIFY = 128;  // tokens checked against the CPU reference

    printf("MoE layer: %d tokens, D=%d, %d experts (FFN %d -> %d -> %d), top-1\n\n",
           N, D, E, D, H, D);

    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));
    float one = 1.f, zero = 0.f;

    // ── Host data ─────────────────────────────────────────────────────────────
    float* h_X  = (float*)malloc((size_t)N * D * sizeof(float));
    float* h_Wr = (float*)malloc((size_t)D * E * sizeof(float));
    float* h_W1 = (float*)malloc((size_t)E * D * H * sizeof(float));
    float* h_W2 = (float*)malloc((size_t)E * H * D * sizeof(float));
    rand_fill(h_X,  N * D, -1.f, 1.f);
    rand_fill(h_Wr, D * E, -0.3f, 0.3f);
    rand_fill(h_W1, E * D * H, -0.05f, 0.05f);
    rand_fill(h_W2, E * H * D, -0.05f, 0.05f);

    // ── Device buffers ────────────────────────────────────────────────────────
    float *d_X, *d_Wr, *d_logits, *d_gate, *d_Xp, *d_hid, *d_Yp, *d_Y;
    float *d_W1, *d_W2;
    int   *d_eid, *d_counts, *d_base, *d_cursor, *d_slot;
    CUDA_CHECK(cudaMalloc(&d_X,      (size_t)N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Wr,     (size_t)D * E * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_logits, (size_t)N * E * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gate,   (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Xp,     (size_t)N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_hid,    (size_t)N * H * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Yp,     (size_t)N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Y,      (size_t)N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W1,     (size_t)E * D * H * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W2,     (size_t)E * H * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_eid,    (size_t)N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_counts, (size_t)E * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_base,   (size_t)E * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cursor, (size_t)E * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_slot,   (size_t)N * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_X,  h_X,  (size_t)N*D*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Wr, h_Wr, (size_t)D*E*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W1, h_W1, (size_t)E*D*H*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W2, h_W2, (size_t)E*H*D*sizeof(float), cudaMemcpyHostToDevice));

    GpuTimer timer; timer.start();

    // ── 1. Router ─────────────────────────────────────────────────────────────
    // logits[N x E] = X[N x D] @ Wr[D x E]
    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
        E, N, D, &one, d_Wr, E, d_X, D, &zero, d_logits, E));
    routerArgmax<<<(N+BLK-1)/BLK, BLK>>>(d_logits, d_eid, d_gate, N, E);

    // ── 2. Group by expert (the local "AllToAll") ─────────────────────────────
    CUDA_CHECK(cudaMemset(d_counts, 0, E * sizeof(int)));
    histKernel<<<(N+BLK-1)/BLK, BLK>>>(d_eid, d_counts, N);
    int h_counts[E], h_base[E];
    CUDA_CHECK(cudaMemcpy(h_counts, d_counts, E*sizeof(int), cudaMemcpyDeviceToHost));
    int acc = 0;
    for (int e = 0; e < E; e++) { h_base[e] = acc; acc += h_counts[e]; }
    CUDA_CHECK(cudaMemcpy(d_base, h_base, E*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_cursor, 0, E * sizeof(int)));
    scatterKernel<<<(N+BLK-1)/BLK, BLK>>>(d_X, d_Xp, d_eid, d_base, d_cursor,
                                          d_slot, N, D);

    // ── 3. One FFN per expert, on its contiguous token slice ─────────────────
    for (int e = 0; e < E; e++) {
        int n_e = h_counts[e];
        if (n_e == 0) continue;
        const float* Xe = d_Xp  + (size_t)h_base[e] * D;
        float*       He = d_hid + (size_t)h_base[e] * H;
        float*       Ye = d_Yp  + (size_t)h_base[e] * D;
        // hid = Xe @ W1_e   ([n_e x D] @ [D x H])
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            H, n_e, D, &one, d_W1 + (size_t)e*D*H, H, Xe, D, &zero, He, H));
        reluKernel<<<((size_t)n_e*H + BLK-1)/BLK, BLK>>>(He, n_e * H);
        // Ye = hid @ W2_e   ([n_e x H] @ [H x D])
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            D, n_e, H, &one, d_W2 + (size_t)e*H*D, D, He, H, &zero, Ye, D));
    }

    // ── 4. Ungroup + gate ─────────────────────────────────────────────────────
    gatherKernel<<<(N+BLK-1)/BLK, BLK>>>(d_Yp, d_Y, d_slot, d_gate, N, D);
    CUDA_CHECK(cudaDeviceSynchronize());
    float ms = timer.stop_ms();

    section("MoE forward complete");
    printf("  route -> group -> %d expert GEMM pairs -> ungroup: %.3f ms\n", E, ms);

    // ── Verify against a per-token CPU reference ──────────────────────────────
    section("Verification (first tokens vs CPU reference)");
    {
        float* h_Y    = (float*)malloc((size_t)N * D * sizeof(float));
        int*   h_eid  = (int*)  malloc((size_t)N * sizeof(int));
        float* h_gate = (float*)malloc((size_t)N * sizeof(float));
        CUDA_CHECK(cudaMemcpy(h_Y,    d_Y,    (size_t)N*D*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_eid,  d_eid,  (size_t)N*sizeof(int),     cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_gate, d_gate, (size_t)N*sizeof(float),   cudaMemcpyDeviceToHost));

        float* hid = (float*)malloc((size_t)H * sizeof(float));
        float* ref = (float*)malloc((size_t)D * sizeof(float));
        float err = 0.f;
        for (int t = 0; t < N_VERIFY; t++) {
            int e = h_eid[t];
            const float* x  = h_X + (size_t)t * D;
            const float* W1 = h_W1 + (size_t)e * D * H;
            const float* W2 = h_W2 + (size_t)e * H * D;
            for (int j = 0; j < H; j++) {
                float s = 0.f;
                for (int k = 0; k < D; k++) s += x[k] * W1[k*H + j];
                hid[j] = s > 0.f ? s : 0.f;
            }
            for (int j = 0; j < D; j++) {
                float s = 0.f;
                for (int k = 0; k < H; k++) s += hid[k] * W2[k*D + j];
                ref[j] = h_gate[t] * s;
            }
            err = fmaxf(err, max_abs_diff(ref, h_Y + (size_t)t*D, D));
        }
        printf("  %d tokens re-computed through their expert on the CPU\n", N_VERIFY);
        printf("  max diff: %.2e  %s\n", err,
               err < 1e-3f ? "PASS -- grouping + gating are exact" : "FAIL");
        free(h_Y); free(h_eid); free(h_gate); free(hid); free(ref);
    }

    // ── Load imbalance ────────────────────────────────────────────────────────
    section("Load imbalance: tokens per expert");
    {
        int mx = 0;
        for (int e = 0; e < E; e++) if (h_counts[e] > mx) mx = h_counts[e];
        double mean = (double)N / E;
        for (int e = 0; e < E; e++) {
            int bar = (int)(40.0 * h_counts[e] / mx);
            printf("  expert %d %5d  ", e, h_counts[e]);
            for (int j = 0; j < bar; j++) putchar('#');
            printf("\n");
        }
        printf("\n  mean %.0f tokens/expert, max %d -> imbalance factor %.2fx\n",
               mean, mx, mx / mean);
        printf("  With expert parallelism, the step waits for the HOT expert:\n");
        printf("  the whole batch runs at the speed of the most popular one.\n");
        printf("  (Training adds an auxiliary load-balancing loss to fight this.)\n");
    }

    // ── Capacity factor ───────────────────────────────────────────────────────
    section("Capacity factor: bounding the worst case");
    {
        printf("  capacity = cf * N/E tokens per expert; overflow is dropped\n");
        printf("  (dropped tokens pass through the residual connection only)\n\n");
        printf("  %-8s %-12s %-12s\n", "cf", "capacity", "dropped");
        float cfs[] = {1.0f, 1.25f, 1.5f, 2.0f};
        for (int c = 0; c < 4; c++) {
            int capacity = (int)(cfs[c] * N / E);
            int dropped = 0;
            for (int e = 0; e < E; e++)
                if (h_counts[e] > capacity) dropped += h_counts[e] - capacity;
            printf("  %-8.2f %-12d %5d (%4.1f%%)\n",
                   cfs[c], capacity, dropped, 100.0 * dropped / N);
        }
        printf("\n  cf trades memory/compute headroom against dropped tokens.\n");
        printf("  Fixed capacity also makes AllToAll message sizes STATIC --\n");
        printf("  which is exactly what the communication layer wants.\n");
    }

    // ── The sparsity ledger ───────────────────────────────────────────────────
    section("Why MoE at all: the parameter/FLOP ledger");
    {
        double params_dense = (double)D*H + (double)H*D;        // one FFN
        double params_moe   = E * params_dense;                  // E FFNs
        double flops_tok    = 2.0 * params_dense;                // top-1: 1 expert
        printf("  dense FFN params : %.1f M     MoE params: %.1f M (%dx)\n",
               params_dense/1e6, params_moe/1e6, E);
        printf("  FLOPs per token  : %.1f M in BOTH cases (top-1 activates 1 expert)\n",
               flops_tok/1e6);
        printf("  -> %dx the parameters at ~1x the compute per token.\n", E);
        printf("  The price: %dx the weight MEMORY, plus two AllToAlls per\n", E);
        printf("  layer once experts live on different GPUs (program 04).\n");
    }

    // Cleanup
    cublasDestroy(cublas);
    cudaFree(d_X); cudaFree(d_Wr); cudaFree(d_logits); cudaFree(d_gate);
    cudaFree(d_Xp); cudaFree(d_hid); cudaFree(d_Yp); cudaFree(d_Y);
    cudaFree(d_W1); cudaFree(d_W2);
    cudaFree(d_eid); cudaFree(d_counts); cudaFree(d_base);
    cudaFree(d_cursor); cudaFree(d_slot);
    free(h_X); free(h_Wr); free(h_W1); free(h_W2);
    return 0;
}
