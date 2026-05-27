// vec_add.cu — Post 2: Your First CUDA Kernels
//
// Demonstrates:
//   • The CUDA thread/block/grid model
//   • cudaMalloc / cudaMemcpy / cudaFree
//   • A simple elementwise kernel
//   • GPU timing with CUDA events
//   • Verification against a CPU reference

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/timer.cuh"

// ─────────────────────────────────────────────────────────────────────────────
// Kernel
// ─────────────────────────────────────────────────────────────────────────────

// Each thread computes exactly one output element.
// The indexing formula maps the flat global thread ID to an array position:
//   global_id = blockIdx.x * blockDim.x + threadIdx.x
//
// The bounds check (i < n) handles the case where n is not a multiple of
// blockDim.x — the last block may contain threads with out-of-range IDs.

__global__ void vecAddKernel(const float* __restrict__ a,
                              const float* __restrict__ b,
                              float*       __restrict__ c,
                              int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// ─────────────────────────────────────────────────────────────────────────────
// CPU reference (for correctness checking)
// ─────────────────────────────────────────────────────────────────────────────

void vecAddCPU(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; i++) c[i] = a[i] + b[i];
}

// ─────────────────────────────────────────────────────────────────────────────
// Verify: compare GPU and CPU results element-wise
// ─────────────────────────────────────────────────────────────────────────────

bool verify(const float* gpu, const float* cpu, int n, float tol = 1e-5f) {
    for (int i = 0; i < n; i++) {
        if (fabsf(gpu[i] - cpu[i]) > tol) {
            fprintf(stderr, "MISMATCH at i=%d: GPU=%.6f CPU=%.6f\n",
                    i, gpu[i], cpu[i]);
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

    // Problem size — override with first command-line argument
    int n = (argc > 1) ? atoi(argv[1]) : (1 << 24);  // 16 M elements default
    printf("Vector length : %d  (%.1f MB per array)\n\n",
           n, (float)n * sizeof(float) / 1e6f);

    // ── Host allocations ────────────────────────────────────────────────────
    float* h_a   = (float*)malloc(n * sizeof(float));
    float* h_b   = (float*)malloc(n * sizeof(float));
    float* h_c   = (float*)malloc(n * sizeof(float));   // GPU result (copied back)
    float* h_ref = (float*)malloc(n * sizeof(float));   // CPU reference

    // Initialise inputs
    for (int i = 0; i < n; i++) {
        h_a[i] = (float)i * 0.001f;
        h_b[i] = (float)(n - i) * 0.001f;
    }

    // ── Device allocations ──────────────────────────────────────────────────
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, n * sizeof(float)));

    // ── Host → Device transfers ─────────────────────────────────────────────
    CUDA_CHECK(cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice));

    // ── Launch configuration ────────────────────────────────────────────────
    // 256 threads/block is a good default: multiple of warp size (32), and
    // leaves enough headroom to maintain high occupancy on most GPUs.
    int blockSize = 256;
    // Ceiling division: enough blocks to cover all n elements.
    int gridSize  = (n + blockSize - 1) / blockSize;

    printf("Launch config : <<<%d blocks, %d threads/block>>>\n\n",
           gridSize, blockSize);

    // ── Warm-up run (not timed) ─────────────────────────────────────────────
    // JIT compilation and driver initialisation happen on the first launch.
    // Always warm up before benchmarking.
    vecAddKernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ── Timed run ───────────────────────────────────────────────────────────
    const int REPEATS = 10;
    GpuTimer timer;
    timer.start();
    for (int r = 0; r < REPEATS; r++)
        vecAddKernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    float ms = timer.stop_ms() / REPEATS;

    // ── Device → Host ────────────────────────────────────────────────────────
    CUDA_CHECK(cudaMemcpy(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost));

    // ── CPU reference ────────────────────────────────────────────────────────
    vecAddCPU(h_a, h_b, h_ref, n);

    // ── Results ──────────────────────────────────────────────────────────────
    bool ok = verify(h_c, h_ref, n);
    printf("Correctness   : %s\n", ok ? "PASS" : "FAIL");

    // Effective memory bandwidth:
    //   We read 2 arrays and write 1 → 3 * n * sizeof(float) bytes
    double bw_gb_s = 3.0 * n * sizeof(float) / (ms * 1e-3) / 1e9;
    printf("Kernel time   : %.3f ms\n", ms);
    printf("Bandwidth     : %.1f GB/s\n\n", bw_gb_s);

    // ── Cleanup ──────────────────────────────────────────────────────────────
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c); free(h_ref);
    return ok ? 0 : 1;
}
