// 04_2d_mesh.cu  --  Post 9: Data and Tensor Parallelism
//
// Real large-scale training combines parallelism strategies. The simplest
// combination is a 2D device mesh: tensor-parallel within a node, data-parallel
// across nodes.
//
// Picture 16 GPUs arranged as a 4x4 mesh:
//
//        TP axis (within node, NVLink) ->
//      +-------+-------+-------+-------+
//   DP | G0    | G1    | G2    | G3    |  <- node 0: TP group {G0..G3}
//   |  +-------+-------+-------+-------+
//   v  | G4    | G5    | G6    | G7    |  <- node 1: TP group {G4..G7}
//      +-------+-------+-------+-------+
//      | G8    | G9    | G10   | G11   |  <- node 2
//      +-------+-------+-------+-------+
//      | G12   | G13   | G14   | G15   |  <- node 3
//      +-------+-------+-------+-------+
//
//   - Each ROW is a tensor-parallel group of TP=4 GPUs (shares a layer)
//   - Each COLUMN is a data-parallel group of DP=4 GPUs (shares a batch shard)
//   - Total GPUs = TP * DP = 16
//
// The two communication patterns happen on different axes:
//   - TP AllReduce: within a row (every layer, must be fast -> NVLink)
//   - DP AllReduce: down a column (once per step, can be slower -> InfiniBand)
//
// This program builds the mesh, computes which GPUs talk to which, and
// analyses the communication volume on each axis. It then computes the
// optimal TP/DP split for a given model and hardware.

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/utils.cuh"

int main(int argc, char** argv) {
    print_device_info();

    // ── A: Build and visualise the mesh ───────────────────────────────────────
    section("A: 2D device mesh layout");
    {
        int TP = (argc > 1) ? atoi(argv[1]) : 4;   // tensor-parallel degree
        int DP = (argc > 2) ? atoi(argv[2]) : 4;   // data-parallel degree
        int total = TP * DP;

        printf("  TP=%d (within node, NVLink)   DP=%d (across nodes, IB)\n", TP, DP);
        printf("  Total GPUs = TP x DP = %d\n\n", total);

        // Map GPU global rank -> (dp_rank, tp_rank)
        // rank = dp_rank * TP + tp_rank
        printf("  Mesh (rows = DP groups, columns = TP groups):\n\n");
        printf("        ");
        for (int t = 0; t < TP; t++) printf(" TP%d  ", t);
        printf("\n");
        for (int d = 0; d < DP; d++) {
            printf("  DP%d  ", d);
            for (int t = 0; t < TP; t++) {
                int rank = d * TP + t;
                printf(" G%-3d ", rank);
            }
            printf("\n");
        }

        printf("\n  GPU rank -> coordinates:\n");
        printf("    tp_rank = rank %% TP   (which slice of the layer)\n");
        printf("    dp_rank = rank / TP   (which batch shard)\n");
    }

    // ── B: Communication on each axis ──────────────────────────────────────────
    section("B: Communication volume on each mesh axis");
    {
        int TP = 4;
        double params      = 7e9;        // 7B model
        double layers      = 32;         // transformer layers
        double H           = 4096;       // hidden dim
        double Bsz         = 8;          // batch per DP group
        double seq         = 2048;       // sequence length

        // TP AllReduce: 2 per layer (forward + backward), each of size
        // activation = Bsz * seq * H elements
        double tp_msg   = Bsz * seq * H * 2;            // BF16 bytes
        double tp_per_step = tp_msg * 2 * layers;        // 2 collectives x layers

        // DP AllReduce: 1 per step, size = gradient = params (sharded by TP)
        double dp_msg   = params / TP * 2;               // BF16 gradient shard
        double dp_per_step = dp_msg;                     // one AllReduce

        printf("  Model: 7B, %0.f layers, H=%.0f, batch/DP=%.0f, seq=%.0f\n\n",
               layers, H, Bsz, seq);
        printf("  TP axis (within node, every layer):\n");
        printf("    Per-layer AllReduce size : %.1f MB\n", tp_msg / 1e6);
        printf("    Total TP traffic/step    : %.1f GB  (2 x %0.f layers)\n",
               tp_per_step / 1e9, layers);
        printf("    -> MUST be on NVLink (happens 64x per step)\n\n");

        printf("  DP axis (across nodes, once per step):\n");
        printf("    Gradient AllReduce size  : %.1f GB  (params/TP)\n", dp_msg / 1e9);
        printf("    Total DP traffic/step    : %.1f GB  (1 collective)\n",
               dp_per_step / 1e9);
        printf("    -> Can tolerate InfiniBand (happens 1x per step)\n\n");

        printf("  Design rule: the frequent, small TP collectives go on the\n");
        printf("  fast intra-node links; the infrequent, large DP collective\n");
        printf("  goes on the slower inter-node links. Matching communication\n");
        printf("  frequency to link speed is the core of mesh design.\n");
    }

    // ── C: Choosing the TP/DP split ────────────────────────────────────────────
    section("C: Choosing TP and DP for a fixed GPU budget");
    {
        int total_gpus = 64;
        double params  = 70e9;       // 70B model

        printf("  70B model, %d GPUs, 16 bytes/param training state\n", total_gpus);
        printf("  Constraint: model state / TP must fit in 80 GB/GPU\n\n");

        printf("  %-6s %-6s %-18s %-16s\n",
               "TP", "DP", "State/GPU", "Fits 80GB H100?");
        printf("  %-6s %-6s %-18s %-16s\n",
               "----", "----", "------------------", "----------------");

        for (int TP : {1, 2, 4, 8, 16}) {
            if (total_gpus % TP != 0) continue;
            int DP = total_gpus / TP;
            // With ZeRO-1 across DP and TP sharding of params:
            // params sharded by TP, optimizer state sharded by DP
            double param_bytes = params * 2 / TP;               // FP16 params / TP
            double opt_bytes   = params * 12 / (TP * DP);        // opt state / (TP*DP)
            double grad_bytes  = params * 2 / TP;                // grad / TP
            double per_gpu     = (param_bytes + opt_bytes + grad_bytes);
            printf("  %-6d %-6d %14.1f GB   %-16s\n",
                   TP, DP, per_gpu / 1e9,
                   per_gpu < 80e9 ? "yes" : "NO -- OOM");
        }
        printf("\n  Larger TP shards the model more (less memory/GPU) but adds\n");
        printf("  per-layer communication. Pick the smallest TP that fits in\n");
        printf("  memory, then use the remaining GPUs for DP. Here TP=4 or TP=8\n");
        printf("  fits; TP=4 minimises the expensive per-layer TP communication.\n");
    }

    // ── D: The full 3D picture (preview of post 10) ───────────────────────────
    section("D: Beyond 2D -- the third dimension (preview)");
    printf("  This 2D mesh (TP x DP) handles models up to ~tens of billions of\n");
    printf("  parameters. Frontier models (100B-1T+) add a THIRD axis:\n\n");
    printf("    Pipeline parallel (PP): split the LAYERS across GPU groups.\n");
    printf("    Each group owns a contiguous block of layers and passes\n");
    printf("    activations to the next group like a factory assembly line.\n\n");
    printf("  3D parallelism = TP x PP x DP. A 1024-GPU cluster might run:\n");
    printf("    TP=8  (within node, NVLink)\n");
    printf("    PP=8  (across 8 nodes, activations at layer boundaries)\n");
    printf("    DP=16 (16 replicas of the TP x PP group)\n");
    printf("    8 x 8 x 16 = 1024 GPUs\n\n");
    printf("  Pipeline parallelism and its bubble overhead, plus mixture-of-\n");
    printf("  experts routing, are covered in post 10.\n");

    return 0;
}
