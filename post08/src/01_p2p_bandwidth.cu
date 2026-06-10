// 01_p2p_bandwidth.cu  --  Post 8: Multi-GPU Infrastructure
//
// Measures the bandwidth of GPU-to-GPU data movement, which is the foundation
// of every multi-GPU training and inference system.
//
// Three transfer paths, fastest to slowest:
//   1. NVLink P2P:   direct GPU-to-GPU over NVLink (~300-900 GB/s on H100)
//   2. PCIe P2P:     direct GPU-to-GPU over PCIe (~25-60 GB/s)
//   3. Host-staged:  GPU0 -> CPU RAM -> GPU1 (~half of PCIe, two hops)
//
// The program:
//   A. Detects all GPUs and queries the P2P access matrix
//   B. Checks which GPU pairs have NVLink vs PCIe connectivity
//   C. Measures actual bandwidth for each available path
//   D. Falls back to single-GPU H2D/D2H measurement if only 1 GPU present
//
// Why this matters: a ring-AllReduce (program 02) moves 2*(N-1)/N times the
// model size across GPU links every training step. If those links are slow
// PCIe instead of NVLink, communication dominates and scaling collapses.

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "../include/utils.cuh"

// ─────────────────────────────────────────────────────────────────────────────
// Measure unidirectional bandwidth between two device pointers
// ─────────────────────────────────────────────────────────────────────────────
static float measureCopyBandwidth(void* dst, int dst_dev,
                                   void* src, int src_dev,
                                   size_t bytes, int reps) {
    // Warm up
    CUDA_CHECK(cudaMemcpyPeer(dst, dst_dev, src, src_dev, bytes));
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer t; t.start();
    for (int r = 0; r < reps; r++)
        CUDA_CHECK(cudaMemcpyPeer(dst, dst_dev, src, src_dev, bytes));
    float ms = t.stop_ms() / reps;
    return (float)bw_gb_s(bytes, ms);
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    print_device_info();

    int num_gpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));

    const size_t MB    = 256;                       // transfer size in MB
    const size_t bytes = MB * 1024 * 1024;
    const int    REPS  = 20;

    if (num_gpus < 2) {
        // ── Single-GPU fallback: measure H2D / D2H bandwidth ─────────────────
        section("Single GPU detected: measuring H2D / D2H bandwidth");
        printf("  Multi-GPU P2P requires 2+ GPUs. Measuring host transfers instead.\n\n");

        float* h_pinned;
        float* d_buf;
        CUDA_CHECK(cudaMallocHost(&h_pinned, bytes));
        CUDA_CHECK(cudaMalloc(&d_buf, bytes));
        memset(h_pinned, 0, bytes);

        // H2D
        CUDA_CHECK(cudaMemcpy(d_buf, h_pinned, bytes, cudaMemcpyHostToDevice));
        GpuTimer t; t.start();
        for (int r = 0; r < REPS; r++)
            CUDA_CHECK(cudaMemcpy(d_buf, h_pinned, bytes, cudaMemcpyHostToDevice));
        float ms_h2d = t.stop_ms() / REPS;

        // D2H
        t.start();
        for (int r = 0; r < REPS; r++)
            CUDA_CHECK(cudaMemcpy(h_pinned, d_buf, bytes, cudaMemcpyDeviceToHost));
        float ms_d2h = t.stop_ms() / REPS;

        printf("  Transfer size: %zu MB (pinned host memory)\n\n", MB);
        printf("  %-20s %12s\n", "Path", "BW (GB/s)");
        printf("  %-20s %12s\n", "--------------------", "----------");
        printf("  %-20s %12.1f\n", "Host -> Device", bw_gb_s(bytes, ms_h2d));
        printf("  %-20s %12.1f\n", "Device -> Host", bw_gb_s(bytes, ms_d2h));
        printf("\n  These transfers go over PCIe (or NVLink-C2C on Grace Hopper).\n");
        printf("  On a typical PCIe 4.0 x16 link: expect ~25 GB/s.\n");
        printf("  On PCIe 5.0 x16: expect ~50 GB/s.\n\n");
        printf("  To see multi-GPU P2P bandwidth, run on a node with 2+ GPUs.\n");
        printf("  The ring-AllReduce in program 02 will simulate the algorithm\n");
        printf("  on a single GPU so you can still see how it works.\n");

        cudaFree(d_buf);
        CUDA_CHECK(cudaFreeHost(h_pinned));
        return 0;
    }

    // ── Multi-GPU: query P2P access matrix ────────────────────────────────────
    section("Peer-to-peer (P2P) access matrix");
    printf("  Can GPU [row] directly access GPU [col]'s memory?\n\n");
    printf("       ");
    for (int j = 0; j < num_gpus; j++) printf("GPU%d  ", j);
    printf("\n");

    // Track which pairs support P2P
    bool* p2p = (bool*)calloc(num_gpus * num_gpus, sizeof(bool));
    for (int i = 0; i < num_gpus; i++) {
        printf("  GPU%d ", i);
        for (int j = 0; j < num_gpus; j++) {
            if (i == j) { printf(" --   "); continue; }
            int can = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&can, i, j));
            p2p[i * num_gpus + j] = (can != 0);
            printf("  %s  ", can ? "Y" : "n");
        }
        printf("\n");
    }
    printf("\n  Y = direct P2P (NVLink or PCIe). n = must stage through host.\n");

    // ── Detect NVLink vs PCIe for connected pairs ─────────────────────────────
    section("Link type detection (NVLink vs PCIe)");
    for (int i = 0; i < num_gpus; i++) {
        for (int j = i + 1; j < num_gpus; j++) {
            int perf_rank = 0, atomics = 0;
            // p2pAttr: lower performance rank = faster link
            cudaDeviceGetP2PAttribute(&perf_rank,
                cudaDevP2PAttrPerformanceRank, i, j);
            cudaDeviceGetP2PAttribute(&atomics,
                cudaDevP2PAttrNativeAtomicSupported, i, j);
            printf("  GPU%d <-> GPU%d: performance_rank=%d  native_atomics=%s\n",
                   i, j, perf_rank, atomics ? "yes" : "no");
        }
    }
    printf("\n  performance_rank 0 = highest bandwidth link (typically NVLink).\n");
    printf("  Higher rank = lower bandwidth (typically PCIe).\n");

    // ── Enable P2P and measure bandwidth ──────────────────────────────────────
    section("P2P transfer bandwidth");
    printf("  Transfer size: %zu MB\n\n", MB);

    // Allocate a buffer on each GPU
    float** d_buf = (float**)malloc(num_gpus * sizeof(float*));
    for (int d = 0; d < num_gpus; d++) {
        CUDA_CHECK(cudaSetDevice(d));
        CUDA_CHECK(cudaMalloc(&d_buf[d], bytes));
        CUDA_CHECK(cudaMemset(d_buf[d], 0, bytes));
    }

    printf("  %-18s %14s %12s\n", "Path", "BW (GB/s)", "Type");
    printf("  %-18s %14s %12s\n", "------------------", "----------", "----------");

    for (int i = 0; i < num_gpus; i++) {
        for (int j = 0; j < num_gpus; j++) {
            if (i == j) continue;

            bool use_p2p = p2p[i * num_gpus + j];
            if (use_p2p) {
                // Enable peer access
                CUDA_CHECK(cudaSetDevice(i));
                cudaError_t e = cudaDeviceEnablePeerAccess(j, 0);
                if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) {
                    // Already enabled or not supported -- clear and continue
                    cudaGetLastError();
                }
            }

            CUDA_CHECK(cudaSetDevice(i));
            float bw = measureCopyBandwidth(d_buf[j], j, d_buf[i], i, bytes, REPS);

            char path[32];
            snprintf(path, sizeof(path), "GPU%d -> GPU%d", i, j);
            int perf_rank = 99;
            if (i < j) cudaDeviceGetP2PAttribute(&perf_rank,
                cudaDevP2PAttrPerformanceRank, i, j);
            printf("  %-18s %14.1f %12s\n", path, bw,
                   use_p2p ? (perf_rank == 0 ? "NVLink?" : "P2P") : "host-staged");
        }
    }

    printf("\n  Interpreting results:\n");
    printf("    >200 GB/s  -> NVLink (excellent for multi-GPU training)\n");
    printf("    25-60 GB/s -> PCIe P2P (acceptable for small models)\n");
    printf("    <25 GB/s   -> host-staged (communication will bottleneck)\n");

    // Cleanup
    for (int d = 0; d < num_gpus; d++) {
        CUDA_CHECK(cudaSetDevice(d));
        cudaFree(d_buf[d]);
    }
    free(d_buf);
    free(p2p);
    return 0;
}
