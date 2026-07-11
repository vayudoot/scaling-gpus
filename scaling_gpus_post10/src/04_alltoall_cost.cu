// 04_alltoall_cost.cu  --  Post 10: Pipeline Parallelism and MoE
//
// When experts live on different GPUs (expert parallelism), program 03's
// local scatter becomes an AllToAll collective: GPU g sends each token to
// the GPU that owns its expert, and a second AllToAll brings results home.
//
// This program:
//   A. SIMULATES an AllToAll across P virtual GPUs (separate buffers on one
//      physical GPU), with uneven per-pair sizes taken from a skewed routing
//      distribution -- and verifies every block lands where it should.
//   B. Measures the cost of the irregularity: total AllToAll time is set by
//      the busiest (src,dst) lane, not the average.
//   C. Applies the alpha-beta cost model to expert parallelism at realistic
//      scale, on NVLink vs InfiniBand.
//   D. Places EP in the 4D parallelism map (DP x TP x PP x EP).

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include "../include/utils.cuh"

// tag every element of a block with an id encoding (src, dst)
__global__ void tagKernel(float* p, int n, float tag) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = tag;
}

int main() {
    print_device_info();
    srand(42);

    const int P = 4;            // virtual GPUs (expert-parallel ranks)
    const int T = 4096;         // tokens per GPU
    const int D = 64;           // floats per token payload
    const int BLK = 256;

    // ── A: build a skewed routing matrix send[src][dst] (tokens) ─────────────
    section("A: AllToAll with routing-dependent message sizes");
    int send[P][P];
    for (int s = 0; s < P; s++) {
        // skew: destination (s+1)%P is "hot" for every source
        int remaining = T;
        for (int d = 0; d < P; d++) {
            double w = (d == (s+1) % P) ? 0.45 : 0.55 / (P - 1);
            send[s][d] = (d == P-1) ? remaining : (int)(w * T);
            remaining -= send[s][d];
        }
    }
    printf("  tokens sent from [row] to [col]  (skewed routing):\n\n       ");
    for (int d = 0; d < P; d++) printf("  GPU%d ", d);
    printf("\n");
    for (int s = 0; s < P; s++) {
        printf("  GPU%d ", s);
        for (int d = 0; d < P; d++) printf(" %5d ", send[s][d]);
        printf("\n");
    }

    // per-GPU send buffers (grouped by destination) and recv buffers
    float* d_send[P]; float* d_recv[P];
    for (int g = 0; g < P; g++) {
        CUDA_CHECK(cudaMalloc(&d_send[g], (size_t)T * D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_recv[g], (size_t)2 * T * D * sizeof(float)));
    }
    // tag each (src,dst) block: value = 100*src + dst
    for (int s = 0; s < P; s++) {
        int off = 0;
        for (int d = 0; d < P; d++) {
            int n = send[s][d] * D;
            tagKernel<<<(n+BLK-1)/BLK, BLK>>>(d_send[s] + (size_t)off,
                                              n, (float)(100*s + d));
            off += n;
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // recv offsets: recv[d] gathers blocks from src 0..P-1 in order
    int recv_off[P][P];   // recv_off[d][s]
    for (int d = 0; d < P; d++) {
        int off = 0;
        for (int s = 0; s < P; s++) { recv_off[d][s] = off; off += send[s][d] * D; }
    }

    // the AllToAll: P*P block copies (device-to-device here; NCCL over
    // NVLink/IB in the real thing -- the traffic pattern is identical)
    GpuTimer timer; timer.start();
    for (int s = 0; s < P; s++) {
        int off = 0;
        for (int d = 0; d < P; d++) {
            int n = send[s][d] * D;
            CUDA_CHECK(cudaMemcpyAsync(d_recv[d] + (size_t)recv_off[d][s],
                                       d_send[s] + (size_t)off,
                                       (size_t)n * sizeof(float),
                                       cudaMemcpyDeviceToDevice));
            off += n;
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    float ms = timer.stop_ms();

    // verify every block landed with the right tag
    bool ok = true;
    float* h_chk = (float*)malloc((size_t)2 * T * D * sizeof(float));
    for (int d = 0; d < P && ok; d++) {
        int total = recv_off[d][P-1] + send[P-1][d] * D;
        CUDA_CHECK(cudaMemcpy(h_chk, d_recv[d], (size_t)total * sizeof(float),
                              cudaMemcpyDeviceToHost));
        for (int s = 0; s < P && ok; s++) {
            float want = (float)(100*s + d);
            int   n    = send[s][d] * D;
            for (int i = 0; i < n; i += 97)    // stride-sample the block
                if (h_chk[recv_off[d][s] + i] != want) ok = false;
        }
    }
    size_t total_bytes = 0;
    for (int s = 0; s < P; s++)
        for (int d = 0; d < P; d++)
            if (s != d) total_bytes += (size_t)send[s][d] * D * sizeof(float);
    printf("\n  %d x %d block copies, %.1f MB cross-GPU traffic: %.3f ms\n",
           P, P, total_bytes / 1e6, ms);
    printf("  every (src,dst) block verified in place: %s\n",
           ok ? "PASS" : "FAIL");

    // ── B: the straggler effect ───────────────────────────────────────────────
    section("B: the busiest lane sets the pace");
    {
        // per-destination inbound volume decides when that GPU can start
        printf("  inbound tokens per GPU (must ALL arrive before experts run):\n");
        int worst = 0;
        for (int d = 0; d < P; d++) {
            int inb = 0;
            for (int s = 0; s < P; s++) inb += send[s][d];
            worst = std::max(worst, inb);
            printf("    GPU%d: %5d\n", d, inb);
        }
        double mean = (double)T;   // uniform would be T per GPU
        printf("\n  worst / uniform = %.2fx -> the AllToAll (and the expert\n",
               worst / mean);
        printf("  compute after it) finishes %.0f%% later than a balanced one.\n",
               100.0 * (worst / mean - 1.0));
        printf("  This is the load-imbalance tax from program 03, now as traffic.\n");
    }

    // ── C: alpha-beta cost of EP AllToAll at scale ────────────────────────────
    section("C: expert-parallel AllToAll cost (per MoE layer = 2 of these)");
    {
        double alpha = 2e-6;                    // per-message latency
        double tokens = 8.0 * 2048;             // batch 8 x seq 2048
        double Dmodel = 4096, bytes_tok = Dmodel * 2;   // BF16 activations
        printf("  batch of %.0f tokens, D=%.0f, BF16 -> %.1f MB of activations\n\n",
               tokens, Dmodel, tokens * bytes_tok / 1e6);
        printf("  %-6s %-14s %-16s %-16s\n", "EP", "bytes/GPU", "NVLink 900GB/s", "IB 50GB/s");
        for (int ep : {2, 4, 8, 16, 64}) {
            // each GPU holds tokens/ep tokens; fraction (ep-1)/ep leaves it
            double out_bytes = (tokens / ep) * bytes_tok * (ep - 1) / ep;
            double t_nvl = alpha * (ep - 1) + out_bytes / 900e9;
            double t_ib  = alpha * (ep - 1) + out_bytes / 50e9;
            printf("  %-6d %10.2f MB   %11.3f ms   %11.3f ms\n",
                   ep, out_bytes / 1e6, t_nvl * 1e3, t_ib * 1e3);
        }
        printf("\n  x2 AllToAlls per MoE layer, x num_layers per step. On IB the\n");
        printf("  collective quickly rivals the expert compute itself -- which is\n");
        printf("  why EP is kept inside the NVLink domain whenever it fits.\n");
    }

    // ── D: where EP sits in the 4D map ────────────────────────────────────────
    section("D: the 4D parallelism map (DP x TP x PP x EP)");
    printf("  %-6s %-26s %-28s\n", "axis", "what it splits", "communication & placement");
    printf("  %-6s %-26s %-28s\n", "----", "--------------------------",
           "----------------------------");
    printf("  %-6s %-26s %-28s\n", "DP", "the batch",
           "AllReduce grads, 1x/step -> IB ok");
    printf("  %-6s %-26s %-28s\n", "TP", "each weight matrix",
           "AllReduce act., every layer -> NVLink");
    printf("  %-6s %-26s %-28s\n", "PP", "the layer stack",
           "P2P act., stage edges -> IB ok");
    printf("  %-6s %-26s %-28s\n", "EP", "the experts",
           "AllToAll x2, every MoE layer -> NVLink");
    printf("\n  Example (Mixtral-class, 64 GPUs): EP=8 within each node,\n");
    printf("  DP=8 across nodes; add PP when the layer stack outgrows a node.\n");
    printf("  EP and TP compete for the same fast intra-node links -- most\n");
    printf("  MoE deployments pick EP over TP for the FFN and keep TP for\n");
    printf("  attention, which is exactly how Mixtral-style serving works.\n");

    for (int g = 0; g < P; g++) { cudaFree(d_send[g]); cudaFree(d_recv[g]); }
    free(h_chk);
    return 0;
}
