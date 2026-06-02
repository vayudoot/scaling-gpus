// 01_softmax.cu  —  Post 6: Attention on One GPU
//
// Softmax is at the heart of the attention mechanism. Getting it right
// requires solving three problems simultaneously:
//   1. Numerical overflow: exp(x) overflows float32 for x > 88
//   2. Two-pass dependency: you can't normalise until you've seen the max
//   3. Parallelism: efficient reduction across all 32 threads in a warp
//
// This file implements and benchmarks four versions:
//   A. Naive 3-pass: find max, compute exp-sum, normalise — 3 HBM reads
//   B. Stable 3-pass: subtract max before exp — correct but still 3 passes
//   C. Online softmax: mathematically identical to B but in ONE pass over data
//      Maintains a running (max, sum) pair that is rescaled as new values arrive.
//      This is the key building block that makes FlashAttention possible.
//   D. Parallel warp-level online softmax: uses __shfl_xor_sync for
//      register-level reductions — no shared memory, maximum throughput.
//
// The online algorithm update rule:
//   When we see a new value x_{i+1} after having accumulated (m, d):
//     m_new = max(m_old, x_{i+1})
//     d_new = d_old * exp(m_old - m_new) + exp(x_{i+1} - m_new)
//   The exp(m_old - m_new) rescaling term corrects the old sum for the
//   new reference point. Since m_new >= m_old, this exponent is <= 0 (safe).
//
// Why online softmax matters:
//   Standard softmax over a row of length N needs 2 passes (find max, then
//   normalise). With online softmax it collapses to 1 pass. This means you can
//   tile the computation: process the row in chunks, combining partial
//   (m, d) statistics from each chunk. FlashAttention does exactly this with
//   Q, K, V tiles — the NxN attention matrix is never materialised in HBM.

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "../include/utils.cuh"

// ─────────────────────────────────────────────────────────────────────────────
// A: Naive softmax — overflows for large inputs
// ─────────────────────────────────────────────────────────────────────────────
// Each block handles one row of length N_COLS.
// Pass 1: compute sum of exp(x).  Pass 2: divide each element.
// Bug: exp(x) overflows float32 if x > ~88. Real attention scores are often
// in the hundreds, so this produces NaN without any warning.

__global__ void softmaxNaive(const float* __restrict__ in,
                              float*       __restrict__ out,
                              int N_COLS) {
    int row  = blockIdx.x;
    int tid  = threadIdx.x;
    const float* row_in  = in  + row * N_COLS;
    float*       row_out = out + row * N_COLS;

    // Pass 1: sum of exp(x)  — no max subtraction, will overflow for large x
    float local_sum = 0.f;
    for (int c = tid; c < N_COLS; c += blockDim.x)
        local_sum += expf(row_in[c]);

    // Reduce sum across threads using shared memory
    extern __shared__ float smem[];
    smem[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid+s];
        __syncthreads();
    }
    float total = smem[0];

    // Pass 2: normalise
    for (int c = tid; c < N_COLS; c += blockDim.x)
        row_out[c] = expf(row_in[c]) / total;
}

// ─────────────────────────────────────────────────────────────────────────────
// B: Stable 3-pass softmax — subtract max before exp
// ─────────────────────────────────────────────────────────────────────────────
// softmax(x - max(x)) == softmax(x) because the constant cancels in the ratio.
// Three passes: find max, compute shifted exp-sum, normalise.
// Correct — but three reads of the row from HBM.

__global__ void softmaxStable3Pass(const float* __restrict__ in,
                                    float*       __restrict__ out,
                                    int N_COLS) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* row_in  = in  + row * N_COLS;
    float*       row_out = out + row * N_COLS;
    extern __shared__ float smem[];

    // Pass 1: find row max
    float local_max = -FLT_MAX;
    for (int c = tid; c < N_COLS; c += blockDim.x)
        local_max = fmaxf(local_max, row_in[c]);
    smem[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid+s]);
        __syncthreads();
    }
    float row_max = smem[0];
    __syncthreads();

    // Pass 2: shifted exp and sum
    float local_sum = 0.f;
    for (int c = tid; c < N_COLS; c += blockDim.x)
        local_sum += expf(row_in[c] - row_max);
    smem[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid+s];
        __syncthreads();
    }
    float total = smem[0];

    // Pass 3: normalise
    for (int c = tid; c < N_COLS; c += blockDim.x)
        row_out[c] = expf(row_in[c] - row_max) / total;
}

// ─────────────────────────────────────────────────────────────────────────────
// C: Online softmax — one pass, same result as B
// ─────────────────────────────────────────────────────────────────────────────
// Each thread maintains its own (running_max, running_sum) pair.
// After processing all elements, threads reduce their partial statistics.
// Then a second pass normalises with the final (max, sum).
//
// The rescaling rule when max changes:
//   new_sum = old_sum * exp(old_max - new_max) + exp(x - new_max)
//
// This requires exactly 2N reads instead of 3N for the stable version.
// In FlashAttention, this collapses further: the normalisation is deferred
// until after all K/V tiles are processed, so the output is computed inline.

__global__ void softmaxOnline(const float* __restrict__ in,
                               float*       __restrict__ out,
                               int N_COLS) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* row_in  = in  + row * N_COLS;
    float*       row_out = out + row * N_COLS;
    extern __shared__ float smem[];
    float* smem_max = smem;
    float* smem_sum = smem + blockDim.x;

    // Pass 1: each thread accumulates its own (running_max, running_sum)
    float running_max = -FLT_MAX;
    float running_sum = 0.f;

    for (int c = tid; c < N_COLS; c += blockDim.x) {
        float x_i   = row_in[c];
        float new_max = fmaxf(running_max, x_i);
        // Rescale old sum when max increases, then add new term
        running_sum = running_sum * expf(running_max - new_max)
                    + expf(x_i - new_max);
        running_max = new_max;
    }

    // Reduce (max, sum) pairs across threads — order-dependent combination
    smem_max[tid] = running_max;
    smem_sum[tid] = running_sum;
    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            float m_a = smem_max[tid],   s_a = smem_sum[tid];
            float m_b = smem_max[tid+s], s_b = smem_sum[tid+s];
            float m_new = fmaxf(m_a, m_b);
            smem_max[tid] = m_new;
            smem_sum[tid] = s_a * expf(m_a - m_new)
                          + s_b * expf(m_b - m_new);
        }
        __syncthreads();
    }
    float global_max = smem_max[0];
    float global_sum = smem_sum[0];

    // Pass 2: normalise (using saved raw values)
    // We still need a second pass here because we didn't store intermediate exps.
    // FlashAttention avoids this second pass by accumulating the output V
    // weighted sum inline during the first pass.
    for (int c = tid; c < N_COLS; c += blockDim.x)
        row_out[c] = expf(row_in[c] - global_max) / global_sum;
}

// ─────────────────────────────────────────────────────────────────────────────
// D: Warp-level online softmax using __shfl_xor_sync
// ─────────────────────────────────────────────────────────────────────────────
// Reduces within a warp using register shuffles — no shared memory needed for
// the reduction itself. Each warp handles one row (N_COLS <= 32).
// For the attention use case this models one head's score row per warp.

__global__ void softmaxWarpLevel(const float* __restrict__ in,
                                  float*       __restrict__ out,
                                  int N_COLS) {
    int row  = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lane = threadIdx.x % 32;

    // Each lane processes one element of the row (assumes N_COLS <= 32)
    float x = (lane < N_COLS) ? in[row * N_COLS + lane] : -FLT_MAX;

    // Warp-level max reduction using butterfly shuffle
    float warp_max = x;
    for (int offset = 16; offset > 0; offset >>= 1)
        warp_max = fmaxf(warp_max, __shfl_xor_sync(0xFFFFFFFF, warp_max, offset));

    // Shifted exp
    float exp_val = (lane < N_COLS) ? expf(x - warp_max) : 0.f;

    // Warp-level sum reduction
    float warp_sum = exp_val;
    for (int offset = 16; offset > 0; offset >>= 1)
        warp_sum += __shfl_xor_sync(0xFFFFFFFF, warp_sum, offset);

    // Write normalised output
    if (lane < N_COLS)
        out[row * N_COLS + lane] = exp_val / warp_sum;
}

// ─────────────────────────────────────────────────────────────────────────────
// CPU reference
// ─────────────────────────────────────────────────────────────────────────────
static void cpuSoftmax(const float* in, float* out, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float m = -FLT_MAX;
        for (int c = 0; c < cols; c++) m = fmaxf(m, in[r*cols+c]);
        float s = 0.f;
        for (int c = 0; c < cols; c++) s += expf(in[r*cols+c] - m);
        for (int c = 0; c < cols; c++)
            out[r*cols+c] = expf(in[r*cols+c] - m) / s;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    print_device_info();
    srand(42);

    // N_ROWS = number of attention score rows (one per query token)
    // N_COLS = sequence length (one score per key token)
    const int N_ROWS = (argc > 1) ? atoi(argv[1]) : 4096;
    const int N_COLS = (argc > 2) ? atoi(argv[2]) : 1024;
    printf("Softmax: %d rows x %d cols  (%.1f MB)\n\n",
           N_ROWS, N_COLS, (double)N_ROWS*N_COLS*4/1e6);

    size_t bytes = (size_t)N_ROWS * N_COLS * sizeof(float);
    float* h_in  = (float*)malloc(bytes);
    float* h_ref = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);
    // Use inputs in range [-5, +5] — realistic post-scale attention scores
    rand_fill(h_in, N_ROWS * N_COLS, -5.f, 5.f);

    cpuSoftmax(h_in, h_ref, N_ROWS, N_COLS);

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    const int BLK  = 256;
    const int REPS = 100;
    size_t smem2   = (size_t)BLK * 2 * sizeof(float);   // for online (max+sum)
    size_t smem1   = (size_t)BLK     * sizeof(float);   // for stable/naive


    section("Softmax kernel comparison");
    printf("  %-22s %10s %10s %12s\n",
           "Kernel", "Time(ms)", "BW(GB/s)", "vs stable");
    printf("  %-22s %10s %10s %12s\n",
           "----------------------","--------","--------","----------");

    float ms_stable = 0;

    // A: Naive (overflow-prone)
    {
        // warm up
        softmaxNaive<<<N_ROWS, BLK, smem1>>>(d_in, d_out, N_COLS);
        CUDA_CHECK(cudaDeviceSynchronize());
        GpuTimer t; t.start();
        for (int r = 0; r < REPS; r++)
            softmaxNaive<<<N_ROWS, BLK, smem1>>>(d_in, d_out, N_COLS);
        float ms = t.stop_ms() / REPS;

        CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
        bool has_nan = false;
        for (int i = 0; i < N_ROWS*N_COLS && !has_nan; i++)
            has_nan = isnan(h_out[i]) || isinf(h_out[i]);
        printf("  %-22s  %9.3f  %9.1f  %s\n", "Naive (no max sub)",
               ms, bw_gb_s(2*bytes, ms),
               has_nan ? "PRODUCES NaN! (overflow)" : "OK for small inputs");
    }

    // B: Stable 3-pass
    {
        softmaxStable3Pass<<<N_ROWS, BLK, smem1>>>(d_in, d_out, N_COLS);
        CUDA_CHECK(cudaDeviceSynchronize());
        GpuTimer t; t.start();
        for (int r = 0; r < REPS; r++)
            softmaxStable3Pass<<<N_ROWS, BLK, smem1>>>(d_in, d_out, N_COLS);
        float ms = t.stop_ms() / REPS;
        ms_stable = ms;
        printf("  %-22s  %9.3f  %9.1f  1.00x (baseline)\n",
               "Stable 3-pass", ms, bw_gb_s(3*bytes, ms));
    }

    // C: Online (1 extra pass for normalise — but only 2 HBM reads of in)
    {
        softmaxOnline<<<N_ROWS, BLK, smem2>>>(d_in, d_out, N_COLS);
        CUDA_CHECK(cudaDeviceSynchronize());
        GpuTimer t; t.start();
        for (int r = 0; r < REPS; r++)
            softmaxOnline<<<N_ROWS, BLK, smem2>>>(d_in, d_out, N_COLS);
        float ms = t.stop_ms() / REPS;
        CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
        float err = max_err(h_out, h_ref, N_ROWS*N_COLS);
        printf("  %-22s  %9.3f  %9.1f  %.2fx  (err %.1e)\n",
               "Online 1-pass", ms, bw_gb_s(2*bytes, ms),
               ms_stable/ms, err);
    }

    // D: Warp-level (only for N_COLS <= 32)
    if (N_COLS <= 32) {
        int warps_per_block = BLK / 32;
        int grid = (N_ROWS + warps_per_block - 1) / warps_per_block;
        softmaxWarpLevel<<<grid, BLK>>>(d_in, d_out, N_COLS);
        CUDA_CHECK(cudaDeviceSynchronize());
        GpuTimer t; t.start();
        for (int r = 0; r < REPS; r++)
            softmaxWarpLevel<<<grid, BLK>>>(d_in, d_out, N_COLS);
        float ms = t.stop_ms() / REPS;
        CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
        float err = max_err(h_out, h_ref, N_ROWS*N_COLS);
        printf("  %-22s  %9.3f  %9.1f  %.2fx  (err %.1e)\n",
               "Warp-level shfl", ms, bw_gb_s(2*bytes, ms),
               ms_stable/ms, err);
    }

    section("Why online softmax enables FlashAttention");
    printf("  Standard softmax needs 2 passes over each row:\n");
    printf("    Pass 1: find max  (one full read of the NxN attention matrix)\n");
    printf("    Pass 2: normalise (another full read + write)\n\n");
    printf("  Online softmax maintains running (max, sum) statistics.\n");
    printf("  When new elements arrive, the accumulated sum is rescaled:\n");
    printf("    sum_new = sum_old * exp(max_old - max_new) + exp(x - max_new)\n\n");
    printf("  This means you can process the row in TILES:\n");
    printf("    - Load a tile of K/V into shared memory\n");
    printf("    - Compute attention scores for that tile\n");
    printf("    - Update running (max, sum) and output accumulator\n");
    printf("    - Move to next tile — the NxN matrix never exists in HBM\n\n");
    printf("  FlashAttention = online softmax + tiling of Q, K, V.\n");
    printf("  The HBM traffic drops from O(N^2) to O(N).\n");

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_ref); free(h_out);
    return 0;
}
