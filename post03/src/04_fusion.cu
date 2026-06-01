// 04_fusion.cu — Post 3: GPU Memory: The Real Bottleneck
//
// Demonstrates kernel fusion: combining multiple elementwise operations into
// a single kernel to eliminate intermediate HBM round-trips.
//
// The operation chain: y = ReLU(LayerNorm(x * weight + bias))
// This is a realistic sequence from a transformer MLP block.
//
// Unfused (4 separate kernels):
//   x * weight + bias  →  write to HBM                    (1 read, 1 write)
//   LayerNorm           →  read from HBM, write to HBM     (1 read, 1 write)
//   ReLU                →  read from HBM, write to HBM     (1 read, 1 write)
//   Total HBM traffic: 3 reads + 3 writes = 6 × N × 4 bytes
//
// Fused (1 kernel):
//   All ops in one pass, only the input is read and only the output is written.
//   Total HBM traffic: 1 read + 1 write = 2 × N × 4 bytes
//   → 3× less HBM traffic → ~3× speedup on a memory-bound workload.
//
// This is the same principle that makes FlashAttention faster (Post 6) and
// that torch.compile() automates by identifying fusable op sequences.

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/timer.cuh"

// ─────────────────────────────────────────────────────────────────────────────
// Unfused kernels — one HBM read + write each
// ─────────────────────────────────────────────────────────────────────────────

// Step 1: elementwise scale + bias
__global__ void scaleBiasKernel(const float* __restrict__ in,
                                  const float* __restrict__ w,
                                  const float* __restrict__ b,
                                  float*       __restrict__ out,
                                  int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * w[i] + b[i];
}

// Step 2: LayerNorm (per-row, row = one sequence position with D features)
//   y = (x - mean) / sqrt(var + eps)
// Simplified version: each block handles one row of length D.
__global__ void layerNormKernel(const float* __restrict__ in,
                                 float*       __restrict__ out,
                                 int D, float eps) {
    extern __shared__ float smem[];
    int tid  = threadIdx.x;
    int base = blockIdx.x * D;

    // Accumulate partial sums in shared memory
    float local_sum = 0.f;
    for (int i = tid; i < D; i += blockDim.x)
        local_sum += in[base + i];
    smem[tid] = local_sum;
    __syncthreads();

    // Parallel reduction for mean
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float mean = smem[0] / D;
    __syncthreads();

    // Accumulate variance
    float local_var = 0.f;
    for (int i = tid; i < D; i += blockDim.x) {
        float diff = in[base + i] - mean;
        local_var += diff * diff;
    }
    smem[tid] = local_var;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float inv_std = rsqrtf(smem[0] / D + eps);

    // Normalise and write output to HBM  ← one full write per element
    for (int i = tid; i < D; i += blockDim.x)
        out[base + i] = (in[base + i] - mean) * inv_std;
}

// Step 3: ReLU
__global__ void reluKernel(const float* __restrict__ in,
                            float*       __restrict__ out,
                            int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = fmaxf(0.f, in[i]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Fused kernel — scale + bias + layernorm + relu, one pass over data
// ─────────────────────────────────────────────────────────────────────────────
//
// Reads: x, weight, bias  (once each per row)
// Writes: out             (once per row)
// Intermediate values (scaled, normalised) stay in registers/shared mem.
// No HBM round-trips between operations.

__global__ void fusedScaleBiasLNReLU(const float* __restrict__ x,
                                      const float* __restrict__ w,
                                      const float* __restrict__ b,
                                      float*       __restrict__ out,
                                      int D, float eps) {
    extern __shared__ float smem[];
    int tid  = threadIdx.x;
    int base = blockIdx.x * D;

    // ── Stage 1: scale + bias, keep in registers (no HBM write) ─────────────
    // Each thread scales its elements and holds them locally.
    // We use smem to compute mean / variance across the row.
    float local_sum = 0.f;
    for (int i = tid; i < D; i += blockDim.x) {
        float val = x[base + i] * w[i] + b[i];   // scale+bias
        smem[i] = val;                             // stage in shared mem for reduction
        local_sum += val;
    }

    // ── Stage 2: mean via parallel reduction ─────────────────────────────────
    // Use the tail of smem for thread-level partial sums
    float* psums = smem + D;
    psums[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) psums[tid] += psums[tid + s];
        __syncthreads();
    }
    float mean = psums[0] / D;
    __syncthreads();

    // ── Stage 3: variance ────────────────────────────────────────────────────
    float local_var = 0.f;
    for (int i = tid; i < D; i += blockDim.x) {
        float diff = smem[i] - mean;
        local_var += diff * diff;
    }
    psums[tid] = local_var;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) psums[tid] += psums[tid + s];
        __syncthreads();
    }
    float inv_std = rsqrtf(psums[0] / D + eps);

    // ── Stage 4: normalise + ReLU, single write to HBM ───────────────────────
    for (int i = tid; i < D; i += blockDim.x) {
        float norm = (smem[i] - mean) * inv_std;
        out[base + i] = fmaxf(0.f, norm);         // ← one write to HBM
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Verify: check fused vs unfused produce the same result
// ─────────────────────────────────────────────────────────────────────────────
static bool verify(const float* a, const float* b, int n, float tol = 1e-4f) {
    for (int i = 0; i < n; i++) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > tol) {
            fprintf(stderr, "Mismatch at %d: fused=%.5f unfused=%.5f diff=%.2e\n",
                    i, a[i], b[i], diff);
            return false;
        }
    }
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    print_device_info();

    // Batch of B sequence positions, each with D features (a transformer row)
    int B = (argc > 1) ? atoi(argv[1]) : 4096;   // batch dimension
    int D = (argc > 2) ? atoi(argv[2]) : 1024;   // feature dimension (must be ≤ blockDim)
    int n = B * D;

    printf("Batch (rows)   : %d\n", B);
    printf("Features (cols): %d\n", D);
    printf("Total elements : %d  (%.1f MB)\n\n", n,
           (double)n * sizeof(float) / 1e6);

    // ── Host setup ───────────────────────────────────────────────────────────
    float* h_x   = (float*)malloc(n * sizeof(float));
    float* h_w   = (float*)malloc(D * sizeof(float));
    float* h_b   = (float*)malloc(D * sizeof(float));
    float* h_out_unfused = (float*)malloc(n * sizeof(float));
    float* h_out_fused   = (float*)malloc(n * sizeof(float));

    srand(42);
    for (int i = 0; i < n; i++) h_x[i] = (rand() % 200 - 100) * 0.01f;
    for (int i = 0; i < D; i++) { h_w[i] = 1.0f + (rand()%100)*0.001f;
                                   h_b[i] = (rand()%100 - 50) * 0.001f; }

    // ── Device setup ─────────────────────────────────────────────────────────
    float *d_x, *d_w, *d_b, *d_tmp1, *d_tmp2, *d_out;
    CUDA_CHECK(cudaMalloc(&d_x,    n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w,    D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b,    D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tmp1, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tmp2, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out,  n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, h_w, D * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, D * sizeof(float), cudaMemcpyHostToDevice));

    int block = 256;
    // Shared memory for fused kernel: D floats (staged values) + block floats (partials)
    size_t smem_fused = (D + block) * sizeof(float);
    size_t smem_ln    = block * sizeof(float);

    const float eps = 1e-5f;
    const int REPS = 50;

    // ── Unfused pipeline ─────────────────────────────────────────────────────
    auto run_unfused = [&]() {
        int g = (n + block - 1) / block;
        scaleBiasKernel<<<g, block>>>(d_x, d_w, d_b, d_tmp1, n);
        layerNormKernel<<<B, block, smem_ln>>>(d_tmp1, d_tmp2, D, eps);
        reluKernel<<<g, block>>>(d_tmp2, d_out, n);
    };

    // Warm up
    run_unfused(); CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer t1; t1.start();
    for (int r = 0; r < REPS; r++) run_unfused();
    float ms_unfused = t1.stop_ms() / REPS;

    CUDA_CHECK(cudaMemcpy(h_out_unfused, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));

    // ── Fused kernel ─────────────────────────────────────────────────────────
    auto run_fused = [&]() {
        fusedScaleBiasLNReLU<<<B, block, smem_fused>>>(d_x, d_w, d_b, d_out, D, eps);
    };

    run_fused(); CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer t2; t2.start();
    for (int r = 0; r < REPS; r++) run_fused();
    float ms_fused = t2.stop_ms() / REPS;

    CUDA_CHECK(cudaMemcpy(h_out_fused, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));

    // ── Results ───────────────────────────────────────────────────────────────
    bool ok = verify(h_out_fused, h_out_unfused, n);

    // HBM traffic:
    //   Unfused: scale+bias (2r+1w), layernorm (1r+1w), relu (1r+1w)
    //            = 4 reads + 3 writes = 7 passes over N elements
    //   Fused:   x,w,b read once + out written once = 3 reads + 1 write = 4 passes
    size_t hbm_unfused = (size_t)(4 + 3) * n * sizeof(float);
    size_t hbm_fused   = (size_t)(3 + 1) * n * sizeof(float);

    printf("%-25s %10s %12s %12s\n", "Version", "Time (ms)", "BW (GB/s)", "HBM traffic");
    printf("%-25s %10s %12s %12s\n", "────────────────────────", "─────────",
           "──────────", "───────────");
    printf("%-25s %10.3f %12.1f %8.0f MB\n", "Unfused (3 kernels)",
           ms_unfused, bandwidth_gb_s(hbm_unfused, ms_unfused),
           (double)hbm_unfused / 1e6);
    printf("%-25s %10.3f %12.1f %8.0f MB  (%.1f× less)\n", "Fused   (1 kernel)",
           ms_fused, bandwidth_gb_s(hbm_fused, ms_fused),
           (double)hbm_fused / 1e6,
           (double)hbm_unfused / hbm_fused);
    printf("\nSpeedup : %.2f×\n", ms_unfused / ms_fused);
    printf("Correct : %s\n\n", ok ? "PASS" : "FAIL");

    printf("Why it's faster\n");
    printf("---------------\n");
    printf("Unfused: each kernel reads the full tensor from HBM then writes it back.\n");
    printf("  scale+bias : 1 read + 1 write\n");
    printf("  layernorm  : 1 read + 1 write\n");
    printf("  relu       : 1 read + 1 write\n");
    printf("  = 3 reads + 3 writes = 6 full tensor passes\n\n");
    printf("Fused:   all ops share a single read of x and single write of out.\n");
    printf("         intermediate values live in shared memory / registers.\n");
    printf("  = 1 read + 1 write = 2 full tensor passes\n\n");
    printf("All three operations are memory-bound (AI < 1 FLOPs/byte each).\n");
    printf("Reducing HBM traffic ≈ reducing total kernel time, almost exactly.\n");
    printf("This is why torch.compile() speeds up transformer forward passes.\n");

    cudaFree(d_x); cudaFree(d_w); cudaFree(d_b);
    cudaFree(d_tmp1); cudaFree(d_tmp2); cudaFree(d_out);
    free(h_x); free(h_w); free(h_b);
    free(h_out_unfused); free(h_out_fused);
    return 0;
}
