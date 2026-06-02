// 03_flash_attention.cu  —  Post 6: Attention on One GPU
//
// FlashAttention forward pass implemented from scratch in CUDA.
//
// The key idea: instead of materialising the full N x N attention matrix in
// HBM, we tile Q, K, V and process one tile pair at a time. Each tile fits
// in on-chip shared memory. The attention matrix lives there and is never
// written to HBM.
//
// This requires online softmax (01_softmax.cu) to work: normally you can't
// normalise row i of the attention matrix until you've seen all N key scores
// for that row. Online softmax maintains a running (max, sum) pair that can
// be updated incrementally as each K tile is processed. At the end, one
// division normalises the accumulated output.
//
// Memory traffic comparison (N=2048, d=64, FP32):
//   Naive:        Q,K(128 KB) + write S(16 MB) + read S(16 MB) + write O = ~33 MB
//   FlashAttention: Q,K,V,O = 4 * N*d*4 bytes = 4 * 512 KB = 2 MB total
//   Savings: ~16x less HBM traffic at N=2048, growing as N increases.
//
// This implementation targets readability. Production FlashAttention
// (the flash-attn library) adds:
//   - FP16/BF16 inputs using Tensor Core wmma instructions
//   - Larger tiles (Br=Bc=64 or 128)
//   - Pipelined K/V tile loading with cp.async
//   - Warp-level register tiling for the inner matmul
//   - Separate forward and backward kernels with recomputation

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "../include/utils.cuh"

// Tile dimensions. Total shared memory per block:
//   tileQ  [Br x d]:  Br*d*4 bytes
//   tileK  [Bc x d]:  Bc*d*4 bytes
//   tileV  [Bc x d]:  Bc*d*4 bytes
// For Br=Bc=16, d=64: 3 * 16 * 64 * 4 = 12 KB — well within SM shared mem.
#define BR  16   // tile rows (query)
#define BC  16   // tile cols (key/value)

// ─────────────────────────────────────────────────────────────────────────────
// FlashAttention forward kernel (one head, FP32)
// ─────────────────────────────────────────────────────────────────────────────
//
// Grid:  one block per query tile (N / Br blocks)
// Block: Br x Bc threads (Br=Bc=16 → 256 threads)
//
// Each block:
//   1. Loads its Q tile [Br x d] into shared memory (stays there the whole time)
//   2. Loops over all K/V tiles (N/Bc iterations):
//      a. Load K tile [Bc x d] and V tile [Bc x d] into shared memory
//      b. Compute score tile S[Br x Bc] = Q_tile @ K_tile^T / sqrt(d)
//         (stays in registers — never written to HBM)
//      c. Apply causal mask (S[i,j] = -inf if j > row_offset + i)
//      d. Online softmax update for each of Br output rows:
//           m_new = max(m_old, rowmax(S_col))
//           O    *= exp(m_old - m_new)   (rescale accumulated output)
//           sum  = sum * exp(m_old - m_new) + rowsum(exp(S_col - m_new))
//      e. Accumulate output: O += exp(S - m_new) @ V_tile
//   3. Normalise: O /= sum
//   4. Write O [Br x d] and L [Br] (log-sum-exp) to HBM

__global__ void flashAttentionFwd(
    const float* __restrict__ Q,    // [N x d]
    const float* __restrict__ K,    // [N x d]
    const float* __restrict__ V,    // [N x d]
    float*       __restrict__ O,    // [N x d]  output
    float*       __restrict__ L,    // [N]       log-sum-exp (for backward)
    int N, int d,
    float scale,    // 1 / sqrt(d)
    bool causal)
{
    // Which Q-tile does this block handle?
    int block_row = blockIdx.x;   // tile index: block_row * BR to (block_row+1)*BR - 1
    int row_start = block_row * BR;

    // Thread layout inside the block: threadIdx.x = col index within tile (0..BC-1)
    //                                 threadIdx.y = row index within tile (0..BR-1)
    int tr = threadIdx.y;  // local row index within Q tile
    int tc = threadIdx.x;  // local col index within K/V tile
    int global_row = row_start + tr;  // global sequence position for this thread's row

    // Shared memory layout:
    //   tileQ [BR x d]
    //   tileK [BC x d]
    //   tileV [BC x d]
    extern __shared__ float smem[];
    float* tileQ = smem;                        // [BR x d]
    float* tileK = smem +  BR * d;              // [BC x d]
    float* tileV = smem + (BR + BC) * d;        // [BC x d]

    // Per-row running statistics (one value per row in the Q-tile, held in registers)
    float m_i = -FLT_MAX;  // running max for row tr
    float d_i = 0.f;       // running denominator (rescaled sum)
    // Accumulate output row (d floats per row)
    // We store the output accumulator in registers; each thread owns one row.
    // For simplicity we use a fixed-size array. Works when d <= 128.
    float O_i[128] = {};   // zero-initialised output accumulator
    // Safety check at compile time via static_assert not available without constexpr N,
    // but runtime d is always <= 128 in our benchmarks.

    // ── Load Q tile into shared memory ───────────────────────────────────────
    // Each thread loads one element: thread (tr, tc) loads Q[global_row][tc]
    // We need to load all d elements per row, so we loop over d in steps of BC.
    for (int k = tc; k < d; k += BC) {
        tileQ[tr * d + k] = (global_row < N) ? Q[global_row * d + k] : 0.f;
    }
    __syncthreads();

    // ── Sweep over all K/V tiles ──────────────────────────────────────────────
    int num_col_tiles = (N + BC - 1) / BC;
    for (int col_tile = 0; col_tile < num_col_tiles; col_tile++) {
        int col_start = col_tile * BC;

        // Load K tile: thread (tr, tc) loads K[col_start + tr][tc..tc+step]
        for (int k = tc; k < d; k += BC) {
            int krow = col_start + tr;
            tileK[tr * d + k] = (krow < N) ? K[krow * d + k] : 0.f;
        }
        // Load V tile similarly
        for (int k = tc; k < d; k += BC) {
            int krow = col_start + tr;
            tileV[tr * d + k] = (krow < N) ? V[krow * d + k] : 0.f;
        }
        __syncthreads();

        // ── Compute S tile = Q_tile @ K_tile^T * scale ────────────────────
        // Thread (tr, tc) computes S[tr][tc] — stays in registers
        float s_ij = 0.f;
        for (int k = 0; k < d; k++)
            s_ij += tileQ[tr * d + k] * tileK[tc * d + k];
        s_ij *= scale;

        // ── Causal mask ──────────────────────────────────────────────────
        int global_col = col_start + tc;
        if (causal && global_col > global_row) s_ij = -FLT_MAX;
        if (global_col >= N) s_ij = -FLT_MAX;

        // ── Online softmax update (per row tr) ────────────────────────────
        // Each row needs the max across all BC columns in this tile.
        // We use warp-level shuffle to find the row's max within the tile.
        // Thread (tr, *) holds one element per row — reduce across tc dimension.
        float row_max = s_ij;
        // Horizontal max across the BC=16 threads in the same row (tr)
        // Threads with same tr and different tc are in the same warp (since BC <= warpSize)
        for (int offset = BC/2; offset > 0; offset >>= 1)
            row_max = fmaxf(row_max, __shfl_xor_sync(0xFFFFFFFF, row_max, offset));

        float m_new = fmaxf(m_i, row_max);

        // Compute exp(s_ij - m_new) for this thread's element
        float p_ij = (s_ij == -FLT_MAX) ? 0.f : expf(s_ij - m_new);

        // Sum of p across tc for row tr (row sum)
        float row_sum = p_ij;
        for (int offset = BC/2; offset > 0; offset >>= 1)
            row_sum += __shfl_xor_sync(0xFFFFFFFF, row_sum, offset);

        // Rescale old output and running sum
        float exp_scale = expf(m_i - m_new);
        for (int k = 0; k < d; k++) O_i[k] *= exp_scale;
        d_i = d_i * exp_scale + row_sum;
        m_i = m_new;

        // ── Accumulate output: O_i += p_ij * V_tile[tc][k] ──────────────
        // Thread (tr, tc) contributes p_ij * V[col_start + tc][*] to row tr's output.
        // Each thread in the tc dimension holds a different p value.
        // We need to sum over tc: O_i[k] += sum_tc(p_ij * tileV[tc * d + k])
        // Use shuffle: broadcast p_ij from each tc lane and accumulate.
        for (int other_tc = 0; other_tc < BC; other_tc++) {
            float p = __shfl_sync(0xFFFFFFFF, p_ij, tr * BC + other_tc);
            // Note: this shuffle is within the warp. For BC=16, BR=16:
            // thread layout is 256 threads = 8 warps. Threads with the same
            // tr but different tc are in the same warp iff BC <= 32 (warp size).
            // For BC=16: threads 0..15 share tr=0, threads 16..31 share tr=1, etc.
            // So lane(other_tc) = tr*16 + other_tc is within the same warp. OK.
            for (int k = tc; k < d; k += BC)
                O_i[k] += p * tileV[other_tc * d + k];
        }

        __syncthreads();  // before next tile overwrites tileK and tileV
    }

    // ── Final normalisation and write to HBM ─────────────────────────────────
    if (global_row < N) {
        float inv_d = 1.f / d_i;
        for (int k = tc; k < d; k += BC)
            O[global_row * d + k] = O_i[k] * inv_d;

        // Save log-sum-exp for backward: L[i] = m_i + log(d_i)
        // Only one thread per row needs to write L (tc == 0)
        if (tc == 0)
            L[global_row] = m_i + logf(d_i);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CPU reference (reused from 02_attention_naive.cu — inline here for self-contained build)
// ─────────────────────────────────────────────────────────────────────────────
static void cpuAttention(const float* Q, const float* K, const float* V,
                          float* O, int N, int d, bool causal) {
    float scale = 1.f / sqrtf((float)d);
    float* S = (float*)malloc((size_t)N * N * sizeof(float));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float dot = 0.f;
            for (int k = 0; k < d; k++) dot += Q[i*d+k] * K[j*d+k];
            S[i*N+j] = dot * scale;
        }
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
    for (int i = 0; i < N; i++)
        for (int k = 0; k < d; k++) {
            float acc = 0.f;
            for (int j = 0; j < N; j++) acc += S[i*N+j] * V[j*d+k];
            O[i*d+k] = acc;
        }
    free(S);
}

// ─────────────────────────────────────────────────────────────────────────────
// Naive causal softmax (used in the performance comparison inside main)
// Defined at file scope because __global__ functions cannot be inside structs.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void naiveSoftmaxCausal(float* S, int N_, bool causal) {
    int row = blockIdx.x, tid = threadIdx.x;
    extern __shared__ float sm[];
    float* sm_max = sm;
    float* sm_sum = sm + blockDim.x;
    int cols = causal ? (row + 1) : N_;
    float lm = -FLT_MAX;
    for (int j = tid; j < N_; j += blockDim.x) {
        float v = (j < cols) ? S[row * N_ + j] : -FLT_MAX;
        lm = fmaxf(lm, v);
    }
    sm_max[tid] = lm; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sm_max[tid] = fmaxf(sm_max[tid], sm_max[tid + s]);
        __syncthreads();
    }
    float rm = sm_max[0]; __syncthreads();
    float ls = 0.f;
    for (int j = tid; j < N_; j += blockDim.x) {
        float v = (j < cols) ? expf(S[row * N_ + j] - rm) : 0.f;
        S[row * N_ + j] = v; ls += v;
    }
    sm_sum[tid] = ls; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sm_sum[tid] += sm_sum[tid + s];
        __syncthreads();
    }
    float tot = sm_sum[0];
    for (int j = tid; j < N_; j += blockDim.x) S[row * N_ + j] /= tot;
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    print_device_info();
    srand(42);

    const int D = 64;   // head dimension

    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));

    // ── Correctness check ─────────────────────────────────────────────────────
    section("FlashAttention correctness vs CPU reference");
    {
        const int N = 256;
        size_t bytes = (size_t)N * D * sizeof(float);

        float* h_Q = (float*)malloc(bytes);
        float* h_K = (float*)malloc(bytes);
        float* h_V = (float*)malloc(bytes);
        float* h_O_flash = (float*)malloc(bytes);
        float* h_O_cpu   = (float*)malloc(bytes);

        rand_fill(h_Q, N*D, -0.1f, 0.1f);
        rand_fill(h_K, N*D, -0.1f, 0.1f);
        rand_fill(h_V, N*D, -0.1f, 0.1f);

        float *d_Q, *d_K, *d_V, *d_O, *d_L;
        CUDA_CHECK(cudaMalloc(&d_Q, bytes));
        CUDA_CHECK(cudaMalloc(&d_K, bytes));
        CUDA_CHECK(cudaMalloc(&d_V, bytes));
        CUDA_CHECK(cudaMalloc(&d_O, bytes));
        CUDA_CHECK(cudaMalloc(&d_L, N * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_Q, h_Q, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_K, h_K, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_V, h_V, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_O, 0, bytes));

        // Shared memory: tileQ[BR*D] + tileK[BC*D] + tileV[BC*D]
        size_t smem = (size_t)(BR + BC + BC) * D * sizeof(float);
        int    grid = (N + BR - 1) / BR;
        dim3   blk(BC, BR);

        flashAttentionFwd<<<grid, blk, smem>>>(
            d_Q, d_K, d_V, d_O, d_L, N, D, 1.f/sqrtf((float)D), /*causal=*/true);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_O_flash, d_O, bytes, cudaMemcpyDeviceToHost));
        cpuAttention(h_Q, h_K, h_V, h_O_cpu, N, D, /*causal=*/true);

        float err = max_err(h_O_flash, h_O_cpu, N*D);
        printf("  N=%d d=%d causal=true\n", N, D);
        printf("  Max error vs CPU: %.2e  %s\n\n", err,
               err < 5e-4f ? "PASS" : "FAIL -- check tile logic");

        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
        cudaFree(d_O); cudaFree(d_L);
        free(h_Q); free(h_K); free(h_V); free(h_O_flash); free(h_O_cpu);
    }

    // ── Performance comparison: naive vs FlashAttention ───────────────────────
    section("Performance: naive vs FlashAttention across sequence lengths");
    printf("  %-6s %12s %12s %12s %12s %12s\n",
           "N", "Naive(ms)", "Flash(ms)", "Speedup", "S size", "HBM saved");
    printf("  %-6s %12s %12s %12s %12s %12s\n",
           "------","----------","----------","--------","--------","----------");

    int sizes[] = {512, 1024, 2048, 4096};
    size_t smem_flash = (size_t)(BR + BC + BC) * D * sizeof(float);
    dim3 blk_flash(BC, BR);

    for (int sz : sizes) {
        int N = sz;
        size_t bqkv = (size_t)N * D * sizeof(float);
        size_t bs   = (size_t)N * N * sizeof(float);

        float *d_Q, *d_K, *d_V, *d_O_n, *d_O_f, *d_S, *d_L;
        CUDA_CHECK(cudaMalloc(&d_Q, bqkv)); CUDA_CHECK(cudaMalloc(&d_K, bqkv));
        CUDA_CHECK(cudaMalloc(&d_V, bqkv)); CUDA_CHECK(cudaMalloc(&d_O_n, bqkv));
        CUDA_CHECK(cudaMalloc(&d_O_f, bqkv)); CUDA_CHECK(cudaMalloc(&d_S, bs));
        CUDA_CHECK(cudaMalloc(&d_L, N * sizeof(float)));

        // Init with random values
        {
            float* tmp = (float*)malloc(bqkv);
            rand_fill(tmp, N*D, -0.1f, 0.1f);
            CUDA_CHECK(cudaMemcpy(d_Q, tmp, bqkv, cudaMemcpyHostToDevice));
            rand_fill(tmp, N*D, -0.1f, 0.1f);
            CUDA_CHECK(cudaMemcpy(d_K, tmp, bqkv, cudaMemcpyHostToDevice));
            rand_fill(tmp, N*D, -0.1f, 0.1f);
            CUDA_CHECK(cudaMemcpy(d_V, tmp, bqkv, cudaMemcpyHostToDevice));
            free(tmp);
        }

        float alpha = 1.f/sqrtf((float)D), beta=0.f, one=1.f, zero=0.f;
        const int BLK = 256;
        const int REPS = (N <= 1024) ? 20 : 5;
        GpuTimer t;

        // Naive: Q@K^T → softmax → P@V
        auto run_naive = [&]() {
            cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                N, N, D, &alpha, d_K, D, d_Q, D, &beta, d_S, N);
            naiveSoftmaxCausal<<<N, BLK, 2*BLK*sizeof(float)>>>(d_S, N, true);
            cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                D, N, N, &one, d_V, D, d_S, N, &zero, d_O_n, D);
        };

        // Flash
        auto run_flash = [&]() {
            flashAttentionFwd<<<(N+BR-1)/BR, blk_flash, smem_flash>>>(
                d_Q, d_K, d_V, d_O_f, d_L, N, D,
                1.f/sqrtf((float)D), /*causal=*/true);
        };

        run_naive(); run_flash(); CUDA_CHECK(cudaDeviceSynchronize());

        t.start(); for (int r=0;r<REPS;r++) run_naive(); float ms_n = t.stop_ms()/REPS;
        t.start(); for (int r=0;r<REPS;r++) run_flash(); float ms_f = t.stop_ms()/REPS;

        // HBM traffic saved: naive needs to write+read the N*N S matrix
        // FlashAttention: only Q,K,V,O = 4*N*d*4 bytes total
        size_t naive_hbm  = 2*bqkv + 2*bs + bqkv;  // QK + 2*S + PV + O
        size_t flash_hbm  = 4*bqkv + N*sizeof(float);  // Q,K,V,O,L
        double saved_pct  = 100.0 * (1.0 - (double)flash_hbm / naive_hbm);

        printf("  %-6d %12.2f %12.2f %12.1fx %8zu MB %9.0f%%\n",
               N, ms_n, ms_f, ms_n/ms_f,
               bs / (1024*1024), saved_pct);

        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
        cudaFree(d_O_n); cudaFree(d_O_f); cudaFree(d_S); cudaFree(d_L);
    }

    printf("\n  FlashAttention speedup grows with N because:\n");
    printf("  - Naive attention's N^2 S matrix grows quadratically in HBM traffic\n");
    printf("  - FlashAttention's HBM traffic is O(N) regardless of tile size\n");
    printf("  - At N=4096, naive writes+reads 64 MB per head; Flash writes 0 MB\n\n");

    section("Shared memory usage and tile sizing");
    printf("  Current tile: BR=%d, BC=%d, d=%d\n", BR, BC, D);
    printf("  Shared mem per block: %.1f KB\n",
           (double)(BR + BC + BC) * D * sizeof(float) / 1024);
    printf("  HBM reads per forward pass (Flash, N=2048): %.1f MB\n",
           4.0 * 2048 * D * sizeof(float) / 1e6);
    printf("  HBM reads per forward pass (Naive, N=2048): %.1f MB\n",
           (4.0*2048*D + 2.0*2048*2048) * sizeof(float) / 1e6);
    printf("\n  Larger tiles (BR=BC=64) improve compute/tile-load ratio but\n");
    printf("  require more shared memory (48 KB vs 12 KB here).\n");
    printf("  Production FlashAttention v2 uses 64x64 tiles and pipelines\n");
    printf("  K/V tile loads with cp.async for near-zero load latency.\n");

    CUBLAS_CHECK(cublasDestroy(cublas));
    return 0;
}
