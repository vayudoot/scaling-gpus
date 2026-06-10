// 02_ring_allreduce.cu  --  Post 8: Multi-GPU Infrastructure
//
// Implements the ring-AllReduce algorithm from scratch -- the communication
// pattern at the heart of data-parallel training.
//
// AllReduce: every GPU starts with its own array; after AllReduce, every GPU
// holds the element-wise SUM of all arrays. In data-parallel training, each
// GPU computes gradients on its own data shard, then AllReduce sums them so
// every GPU can apply the same averaged update.
//
// The naive approach (every GPU sends its full array to every other GPU)
// moves O(N * P^2) bytes for P GPUs. Ring-AllReduce moves only O(N) bytes
// per GPU regardless of P -- it is bandwidth-optimal.
//
// Ring-AllReduce has two phases, each with P-1 steps:
//   Phase 1 (reduce-scatter): chunks circulate and accumulate. After P-1
//     steps, each GPU holds the complete sum for exactly one chunk.
//   Phase 2 (all-gather): the completed chunks circulate so every GPU ends
//     up with every completed chunk.
//
// Total data moved per GPU: 2 * (P-1)/P * N -- approaches 2N as P grows,
// independent of P. This is why ring-AllReduce scales.
//
// This program:
//   - Runs a REAL ring-AllReduce across GPUs if 2+ are available
//   - SIMULATES the algorithm on a single GPU otherwise (separate buffers
//     act as "virtual GPUs"), so you can see the chunk movement and verify
//     correctness either way.

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "../include/utils.cuh"

// ─────────────────────────────────────────────────────────────────────────────
// Element-wise add kernel: dst += src
// ─────────────────────────────────────────────────────────────────────────────
__global__ void addKernel(float* __restrict__ dst,
                          const float* __restrict__ src,
                          int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
}

// ─────────────────────────────────────────────────────────────────────────────
// Single-GPU SIMULATION of ring-AllReduce
// ─────────────────────────────────────────────────────────────────────────────
// We model P "virtual GPUs" as P separate device buffers on one physical GPU.
// Copies between them use cudaMemcpy (D2D). This is not fast -- it's a teaching
// tool to show the algorithm produces the correct result and moves the right
// amount of data.
static void simulateRingAllReduce(int P, int N, bool verbose) {
    const int BLK = 256;

    // Each virtual GPU's data buffer [N], split conceptually into P chunks
    float** buf = (float**)malloc(P * sizeof(float*));
    float** h   = (float**)malloc(P * sizeof(float*));

    int chunk = N / P;   // assume N divisible by P for simplicity

    // Initialise: GPU r gets values (r+1) everywhere, so the sum is
    // P*(P+1)/2 in every element -- easy to verify.
    for (int r = 0; r < P; r++) {
        CUDA_CHECK(cudaMalloc(&buf[r], (size_t)N * sizeof(float)));
        h[r] = (float*)malloc((size_t)N * sizeof(float));
        for (int i = 0; i < N; i++) h[r][i] = (float)(r + 1);
        CUDA_CHECK(cudaMemcpy(buf[r], h[r], (size_t)N * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    // Temporary receive buffers (one chunk each)
    float** recv = (float**)malloc(P * sizeof(float*));
    for (int r = 0; r < P; r++)
        CUDA_CHECK(cudaMalloc(&recv[r], (size_t)chunk * sizeof(float)));

    // ── Phase 1: reduce-scatter ───────────────────────────────────────────────
    // In step s, GPU r sends chunk ((r - s) mod P) to GPU (r+1) mod P,
    // which adds it into its own copy of that chunk.
    // After P-1 steps, GPU r holds the full sum of chunk ((r+1) mod P).
    for (int s = 0; s < P - 1; s++) {
        // All sends happen "simultaneously" in a real ring; here we stage them
        for (int r = 0; r < P; r++) {
            int send_chunk = ((r - s) % P + P) % P;
            int dst = (r + 1) % P;
            // Copy chunk from GPU r to recv buffer of GPU dst
            CUDA_CHECK(cudaMemcpy(recv[dst],
                                  buf[r] + send_chunk * chunk,
                                  (size_t)chunk * sizeof(float),
                                  cudaMemcpyDeviceToDevice));
        }
        // Each GPU adds the received chunk into its own
        for (int r = 0; r < P; r++) {
            int recv_chunk = ((r - s - 1) % P + P) % P;
            addKernel<<<(chunk + BLK - 1) / BLK, BLK>>>(
                buf[r] + recv_chunk * chunk, recv[r], chunk);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // ── Phase 2: all-gather ───────────────────────────────────────────────────
    // The completed chunks circulate so every GPU gets every completed chunk.
    for (int s = 0; s < P - 1; s++) {
        for (int r = 0; r < P; r++) {
            int send_chunk = ((r + 1 - s) % P + P) % P;
            int dst = (r + 1) % P;
            CUDA_CHECK(cudaMemcpy(buf[dst] + send_chunk * chunk,
                                  buf[r]   + send_chunk * chunk,
                                  (size_t)chunk * sizeof(float),
                                  cudaMemcpyDeviceToDevice));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // ── Verify: every buffer should hold P*(P+1)/2 in every element ──────────
    float expected = (float)(P * (P + 1) / 2);
    float max_err = 0.f;
    for (int r = 0; r < P; r++) {
        CUDA_CHECK(cudaMemcpy(h[r], buf[r], (size_t)N * sizeof(float),
                              cudaMemcpyDeviceToHost));
        for (int i = 0; i < N; i++)
            max_err = fmaxf(max_err, fabsf(h[r][i] - expected));
    }

    if (verbose) {
        printf("  Virtual GPUs (P) : %d\n", P);
        printf("  Array length (N) : %d  (chunk size = %d)\n", N, chunk);
        printf("  Expected sum     : %.0f  (= P*(P+1)/2)\n", expected);
        printf("  Max error        : %.2e  %s\n", max_err,
               max_err < 1e-3f ? "PASS" : "FAIL");

        // Data movement analysis
        double bytes_per_gpu = 2.0 * (P - 1) / P * N * sizeof(float);
        double naive_bytes   = (double)(P - 1) * N * sizeof(float);
        printf("\n  Bytes moved per GPU (ring)  : %.1f KB  = 2*(P-1)/P * N * 4\n",
               bytes_per_gpu / 1024);
        printf("  Bytes moved per GPU (naive) : %.1f KB  = (P-1) * N * 4\n",
               naive_bytes / 1024);
        printf("  Ring advantage at P=%d      : %.2fx less data\n",
               P, naive_bytes / bytes_per_gpu);
    }

    // Cleanup
    for (int r = 0; r < P; r++) {
        cudaFree(buf[r]); cudaFree(recv[r]); free(h[r]);
    }
    free(buf); free(recv); free(h);
}

// ─────────────────────────────────────────────────────────────────────────────
// Real multi-GPU ring-AllReduce
// ─────────────────────────────────────────────────────────────────────────────
static void realRingAllReduce(int P, int N) {
    const int BLK = 256;
    int chunk = N / P;

    // Enable P2P between ring neighbours
    for (int r = 0; r < P; r++) {
        CUDA_CHECK(cudaSetDevice(r));
        int next = (r + 1) % P;
        int can = 0;
        cudaDeviceCanAccessPeer(&can, r, next);
        if (can) {
            cudaError_t e = cudaDeviceEnablePeerAccess(next, 0);
            if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled)
                cudaGetLastError();
        }
    }

    // Allocate per-GPU buffers
    float** buf  = (float**)malloc(P * sizeof(float*));
    float** recv = (float**)malloc(P * sizeof(float*));
    float** h    = (float**)malloc(P * sizeof(float*));
    for (int r = 0; r < P; r++) {
        CUDA_CHECK(cudaSetDevice(r));
        CUDA_CHECK(cudaMalloc(&buf[r],  (size_t)N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&recv[r], (size_t)chunk * sizeof(float)));
        h[r] = (float*)malloc((size_t)N * sizeof(float));
        for (int i = 0; i < N; i++) h[r][i] = (float)(r + 1);
        CUDA_CHECK(cudaMemcpy(buf[r], h[r], (size_t)N * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    GpuTimer timer;
    CUDA_CHECK(cudaSetDevice(0));
    timer.start();

    // Phase 1: reduce-scatter
    for (int s = 0; s < P - 1; s++) {
        for (int r = 0; r < P; r++) {
            int send_chunk = ((r - s) % P + P) % P;
            int dst = (r + 1) % P;
            CUDA_CHECK(cudaMemcpyPeer(recv[dst], dst,
                                      buf[r] + send_chunk * chunk, r,
                                      (size_t)chunk * sizeof(float)));
        }
        for (int r = 0; r < P; r++) {
            CUDA_CHECK(cudaSetDevice(r));
            int recv_chunk = ((r - s - 1) % P + P) % P;
            addKernel<<<(chunk + BLK - 1) / BLK, BLK>>>(
                buf[r] + recv_chunk * chunk, recv[r], chunk);
        }
        for (int r = 0; r < P; r++) {
            CUDA_CHECK(cudaSetDevice(r));
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    // Phase 2: all-gather
    for (int s = 0; s < P - 1; s++) {
        for (int r = 0; r < P; r++) {
            int send_chunk = ((r + 1 - s) % P + P) % P;
            int dst = (r + 1) % P;
            CUDA_CHECK(cudaMemcpyPeer(buf[dst] + send_chunk * chunk, dst,
                                      buf[r]   + send_chunk * chunk, r,
                                      (size_t)chunk * sizeof(float)));
        }
        for (int r = 0; r < P; r++) {
            CUDA_CHECK(cudaSetDevice(r));
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    CUDA_CHECK(cudaSetDevice(0));
    float ms = timer.stop_ms();

    // Verify
    float expected = (float)(P * (P + 1) / 2);
    float max_err = 0.f;
    for (int r = 0; r < P; r++) {
        CUDA_CHECK(cudaSetDevice(r));
        CUDA_CHECK(cudaMemcpy(h[r], buf[r], (size_t)N * sizeof(float),
                              cudaMemcpyDeviceToHost));
        for (int i = 0; i < N; i++)
            max_err = fmaxf(max_err, fabsf(h[r][i] - expected));
    }

    printf("  Real ring-AllReduce across %d GPUs\n", P);
    printf("  Array length     : %d  (%.1f MB per GPU)\n",
           N, (double)N * sizeof(float) / 1e6);
    printf("  Time             : %.3f ms\n", ms);
    printf("  Max error        : %.2e  %s\n", max_err,
           max_err < 1e-3f ? "PASS" : "FAIL");

    double bytes_moved = 2.0 * (P - 1) / P * N * sizeof(float);
    printf("  Effective BW     : %.1f GB/s per GPU\n", bw_gb_s((size_t)bytes_moved, ms));

    for (int r = 0; r < P; r++) {
        CUDA_CHECK(cudaSetDevice(r));
        cudaFree(buf[r]); cudaFree(recv[r]); free(h[r]);
    }
    free(buf); free(recv); free(h);
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    print_device_info();

    int num_gpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));

    section("Ring-AllReduce algorithm");
    printf("  AllReduce: every GPU ends with the element-wise SUM of all inputs.\n");
    printf("  Used in data-parallel training to average gradients across GPUs.\n\n");
    printf("  Ring algorithm: two phases of P-1 steps each.\n");
    printf("    Phase 1 (reduce-scatter): chunks circulate and accumulate\n");
    printf("    Phase 2 (all-gather): completed chunks circulate to all GPUs\n");
    printf("  Data per GPU: 2*(P-1)/P * N  (approaches 2N, independent of P)\n");

    if (num_gpus >= 2) {
        section("Real multi-GPU ring-AllReduce");
        int P = num_gpus;
        int N = (argc > 1) ? atoi(argv[1]) : (1 << 22);
        N = (N / P) * P;  // round to multiple of P
        realRingAllReduce(P, N);
    } else {
        section("Single GPU: simulating the algorithm with virtual GPUs");
        printf("  No second GPU found -- simulating ring-AllReduce using\n");
        printf("  separate device buffers as virtual GPUs. The algorithm and\n");
        printf("  its correctness are identical; only the physical links differ.\n\n");

        // Show the algorithm at a few "GPU counts"
        for (int P : {2, 4, 8}) {
            int N = 4096 * P;  // divisible by P
            simulateRingAllReduce(P, N, true);
            printf("\n");
        }
    }

    section("Why ring-AllReduce scales");
    printf("  %-6s %-20s %-20s %-12s\n",
           "P", "Naive bytes/GPU", "Ring bytes/GPU", "Ring wins");
    printf("  %-6s %-20s %-20s %-12s\n",
           "------", "--------------------", "--------------------", "----------");
    int N = 1 << 20;  // 1M elements
    for (int P : {2, 4, 8, 16, 64, 256}) {
        double naive = (double)(P - 1) * N * sizeof(float);
        double ring  = 2.0 * (P - 1) / P * N * sizeof(float);
        printf("  %-6d %15.1f MB %15.1f MB %10.1fx\n",
               P, naive / 1e6, ring / 1e6, naive / ring);
    }
    printf("\n  Naive AllReduce: data per GPU grows linearly with P -> doesn't scale.\n");
    printf("  Ring AllReduce: data per GPU is ~2N constant -> scales to thousands.\n");
    printf("  This is why every framework (PyTorch DDP, Horovod) uses ring or\n");
    printf("  tree-based AllReduce via NCCL under the hood.\n");

    return 0;
}
