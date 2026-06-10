// 04_comm_cost_model.cu  --  Post 8: Multi-GPU Infrastructure
//
// A calculator and analysis tool for the communication cost of distributed
// training. No GPUs-to-GPU traffic here -- this program builds the mental
// model for WHEN communication becomes the bottleneck.
//
// The core question for any distributed setup:
//   Does compute time per step exceed communication time per step?
//   If yes -> communication is hidden, you scale well.
//   If no  -> communication dominates, adding GPUs stops helping.
//
// This program computes, for realistic model and hardware configurations:
//   A. The alpha-beta cost model for a single collective
//   B. AllReduce time vs gradient size at different link bandwidths
//   C. The compute/communication ratio for data-parallel training
//   D. The point where you must switch from data to tensor/pipeline parallel
//
// The alpha-beta model for a ring collective:
//   T = alpha * 2*(P-1)  +  beta * 2*(P-1)/P * N
//   where alpha = per-message latency, beta = per-byte transfer time (1/BW),
//   N = message size in bytes, P = number of GPUs.
// Latency (alpha) dominates for small messages; bandwidth (beta) for large.

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/utils.cuh"

// ─────────────────────────────────────────────────────────────────────────────
// Ring AllReduce time under the alpha-beta model
// ─────────────────────────────────────────────────────────────────────────────
// alpha : latency per message (seconds)
// beta  : seconds per byte (= 1 / bandwidth_bytes_per_sec)
// N     : message size in bytes
// P     : number of GPUs
static double ringAllReduceTime(double alpha, double beta, double N, int P) {
    double latency_term   = alpha * 2.0 * (P - 1);
    double bandwidth_term = beta  * 2.0 * (P - 1) / P * N;
    return latency_term + bandwidth_term;
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main() {
    // Note: this is an analysis tool; it doesn't require a specific GPU.
    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);
    printf("Communication cost model (analysis tool)\n");
    printf("Physical GPUs present: %d (not required for this analysis)\n", num_gpus);

    // ── A: The alpha-beta model ───────────────────────────────────────────────
    section("A: Latency vs bandwidth regimes for AllReduce");
    {
        // Typical NVLink values
        double alpha = 2e-6;        // 2 microseconds per message
        double beta  = 1.0 / 300e9; // 300 GB/s -> seconds per byte
        int    P     = 8;

        printf("  Link: alpha=%.0f us latency, beta=1/(%.0f GB/s)\n",
               alpha * 1e6, 1.0 / beta / 1e9);
        printf("  P = %d GPUs\n\n", P);
        printf("  %-14s %-14s %-14s %-14s\n",
               "Msg size", "Latency term", "Bandwidth term", "Dominated by");
        printf("  %-14s %-14s %-14s %-14s\n",
               "----------", "------------", "--------------", "------------");

        size_t sizes[] = {1024, 64*1024, 1024*1024, 64*1024*1024, 1024ull*1024*1024};
        const char* labels[] = {"1 KB", "64 KB", "1 MB", "64 MB", "1 GB"};
        for (int i = 0; i < 5; i++) {
            double N = (double)sizes[i];
            double lat = alpha * 2.0 * (P - 1);
            double bw  = beta * 2.0 * (P - 1) / P * N;
            printf("  %-14s %10.2f us %10.2f us %-14s\n",
                   labels[i], lat * 1e6, bw * 1e6,
                   lat > bw ? "LATENCY" : "BANDWIDTH");
        }
        printf("\n  Small gradients (early layers, biases): latency-bound.\n");
        printf("  Large gradients (big weight matrices): bandwidth-bound.\n");
        printf("  NCCL fuses small gradients into buckets to amortise latency.\n");
    }

    // ── B: AllReduce time vs link bandwidth ───────────────────────────────────
    section("B: AllReduce time for a 7B model gradient at different links");
    {
        double params = 7e9;
        double grad_bytes = params * 2;  // BF16 gradients = 2 bytes/param
        int P = 8;
        double alpha = 2e-6;

        printf("  Model: 7B params, BF16 gradients = %.1f GB\n", grad_bytes / 1e9);
        printf("  P = %d GPUs\n\n", P);
        printf("  %-22s %-14s %-16s\n", "Link", "Bandwidth", "AllReduce time");
        printf("  %-22s %-14s %-16s\n", "----------------------", "----------", "--------------");

        struct { const char* name; double bw; } links[] = {
            {"NVLink 4 (H100)",   900e9},
            {"NVLink 3 (A100)",   600e9},
            {"PCIe 5.0 x16",       64e9},
            {"PCIe 4.0 x16",       32e9},
            {"InfiniBand HDR",     25e9},
            {"Ethernet 100GbE",    12.5e9},
        };
        for (auto& l : links) {
            double beta = 1.0 / l.bw;
            double t = ringAllReduceTime(alpha, beta, grad_bytes, P);
            printf("  %-22s %8.0f GB/s %12.2f ms\n", l.name, l.bw / 1e9, t * 1e3);
        }
        printf("\n  A 7B gradient AllReduce takes ~7 ms on NVLink but ~280 ms\n");
        printf("  on 100GbE. If your compute step is 100 ms, NVLink hides the\n");
        printf("  comm (7ms << 100ms) but Ethernet does not (280ms >> 100ms).\n");
    }

    // ── C: Compute vs communication ratio ─────────────────────────────────────
    section("C: When does data-parallel scaling break down?");
    {
        // For data-parallel training, each step:
        //   compute  = forward + backward FLOPs / GPU throughput
        //   comm     = AllReduce of all gradients
        // Scaling holds while comm < compute (comm can be overlapped/hidden).

        double params      = 7e9;
        double grad_bytes  = params * 2;       // BF16
        double gpu_tflops  = 990e12;           // H100 BF16 dense ~990 TFLOP/s
        double mfu         = 0.4;              // realistic 40% utilisation
        double eff_flops   = gpu_tflops * mfu;
        int    P           = 8;
        double nvlink_bw   = 900e9;
        double alpha       = 2e-6;

        printf("  Model: 7B params  Hardware: 8x H100 NVLink  MFU: 40%%\n\n");
        printf("  %-14s %-16s %-16s %-12s\n",
               "Batch/GPU", "Compute time", "Comm time", "Comm/Compute");
        printf("  %-14s %-16s %-16s %-12s\n",
               "----------", "------------", "------------", "------------");

        // FLOPs per token for forward+backward ~ 6 * params
        double flops_per_token = 6.0 * params;
        double beta = 1.0 / nvlink_bw;
        double comm_time = ringAllReduceTime(alpha, beta, grad_bytes, P);

        for (int batch : {1, 8, 64, 256, 1024}) {
            double compute_flops = flops_per_token * batch;
            double compute_time  = compute_flops / eff_flops;
            double ratio = comm_time / compute_time;
            printf("  %-14d %12.2f ms %12.2f ms %10.2f%s\n",
                   batch, compute_time * 1e3, comm_time * 1e3, ratio,
                   ratio < 0.3 ? "  (well hidden)" :
                   ratio < 1.0 ? "  (marginal)" : "  (comm-bound!)");
        }
        printf("\n  Larger batch per GPU -> more compute per step -> comm more easily\n");
        printf("  hidden. This is why large-batch training scales better.\n");
        printf("  Below batch=8/GPU here, communication starts to dominate.\n");
    }

    // ── D: Choosing a parallelism strategy ────────────────────────────────────
    section("D: When to switch from data to tensor/pipeline parallel");
    {
        printf("  Data parallel breaks down when:\n");
        printf("    - Model doesn't fit on one GPU (need to split the model)\n");
        printf("    - Batch/GPU too small to hide gradient AllReduce\n");
        printf("    - Gradient size so large AllReduce exceeds step compute\n\n");

        printf("  Strategy decision (rough guide):\n");
        printf("  %-28s %-30s\n", "Situation", "Strategy");
        printf("  %-28s %-30s\n", "----------------------------", "------------------------------");
        printf("  %-28s %-30s\n", "Model fits, many GPUs",       "Data parallel (DDP)");
        printf("  %-28s %-30s\n", "Model fits, memory tight",    "ZeRO / FSDP (shard optimizer)");
        printf("  %-28s %-30s\n", "Layer too big for 1 GPU",     "Tensor parallel (split matmul)");
        printf("  %-28s %-30s\n", "Many layers, model too big",  "Pipeline parallel (split layers)");
        printf("  %-28s %-30s\n", "Frontier scale (100B+)",      "3D parallel (DP + TP + PP)");
        printf("\n  Tensor parallel needs the FASTEST links (NVLink) because it\n");
        printf("  communicates inside every layer. Pipeline parallel tolerates\n");
        printf("  slower links because it communicates only at layer boundaries.\n");
        printf("  This is why TP stays within a node (NVLink) and PP crosses\n");
        printf("  nodes (InfiniBand). Covered in detail in posts 9 and 10.\n");
    }

    section("Summary: the communication hierarchy");
    printf("  Within a GPU  : HBM      ~3 TB/s   (post 3)\n");
    printf("  Within a node : NVLink   ~900 GB/s (this post)\n");
    printf("  Between nodes : InfiniBand ~25-50 GB/s (this post)\n");
    printf("  Each tier is ~10-100x slower than the one above. Good distributed\n");
    printf("  design keeps the most frequent communication on the fastest tier.\n");
    return 0;
}
