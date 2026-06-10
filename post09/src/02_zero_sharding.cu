// 02_zero_sharding.cu  --  Post 9: Data and Tensor Parallelism
//
// ZeRO (Zero Redundancy Optimizer) attacks the central weakness of DDP:
// every GPU stores a full copy of the model, gradients, and optimizer state.
// For a 7B model trained with Adam in mixed precision, that is:
//   - Parameters (FP16):       14 GB
//   - Gradients (FP16):        14 GB
//   - FP32 master weights:     28 GB
//   - Adam momentum (FP32):    28 GB
//   - Adam variance (FP32):    28 GB
//   Total: 112 GB per GPU -- exceeds an 80 GB H100 before activations.
//
// ZeRO shards this state across the P data-parallel GPUs instead of
// replicating it. There are three progressive stages:
//   Stage 1: shard the optimizer state (m, v, FP32 master) -> /P
//   Stage 2: also shard the gradients -> /P
//   Stage 3: also shard the parameters -> /P (FSDP in PyTorch)
//
// The trade-off: more sharding saves memory but adds communication.
//   Stage 1/2: same comm as DDP (one reduce-scatter + all-gather = AllReduce)
//   Stage 3: parameters must be all-gathered before each layer's forward and
//            backward, adding communication proportional to model size.
//
// This program:
//   A. Computes the exact memory footprint at each ZeRO stage
//   B. Simulates Stage 1 optimizer-state sharding and verifies the update
//      matches an unsharded optimizer
//   C. Shows the memory-vs-communication trade-off table

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/utils.cuh"

// Adam update for a shard of parameters (each GPU owns 1/P of the params)
__global__ void adamUpdateShard(float* param, const float* grad,
                                 float* m, float* v,
                                 float lr, float b1, float b2, float eps,
                                 float bc1, float bc2, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g  = grad[i];
    float mi = b1 * m[i] + (1.f - b1) * g;
    float vi = b2 * v[i] + (1.f - b2) * g * g;
    m[i] = mi; v[i] = vi;
    param[i] -= lr * (mi / bc1) / (sqrtf(vi / bc2) + eps);
}

int main(int argc, char** argv) {
    print_device_info();
    srand(42);

    // ── A: Memory footprint analysis ──────────────────────────────────────────
    section("A: Training memory footprint per GPU at each ZeRO stage");
    {
        // Mixed-precision Adam memory model (bytes per parameter):
        //   FP16 param: 2, FP16 grad: 2, FP32 master: 4, Adam m: 4, Adam v: 4
        // DDP replicates all 16 bytes on every GPU.
        double bytes_param   = 2;   // FP16 working weights
        double bytes_grad    = 2;   // FP16 gradients
        double bytes_master  = 4;   // FP32 master copy
        double bytes_m       = 4;   // Adam first moment
        double bytes_v       = 4;   // Adam second moment
        double opt_state     = bytes_master + bytes_m + bytes_v;  // 12 bytes
        double total_per_param = bytes_param + bytes_grad + opt_state; // 16

        printf("  Mixed-precision Adam: 16 bytes/param total\n");
        printf("    FP16 param=2, FP16 grad=2, FP32 master=4, Adam m=4, Adam v=4\n\n");

        struct { const char* name; double params; } models[] = {
            {"1.5B", 1.5e9}, {"7B", 7e9}, {"13B", 13e9}, {"70B", 70e9}
        };

        printf("  Per-GPU memory (P=8 GPUs):\n");
        printf("  %-8s %-10s %-10s %-10s %-10s\n",
               "Model", "DDP", "ZeRO-1", "ZeRO-2", "ZeRO-3");
        printf("  %-8s %-10s %-10s %-10s %-10s\n",
               "------", "--------", "--------", "--------", "--------");

        int P = 8;
        for (auto& mdl : models) {
            double n = mdl.params;
            // DDP: everything replicated
            double ddp = n * total_per_param;
            // ZeRO-1: optimizer state sharded (/P), param+grad replicated
            double z1  = n * (bytes_param + bytes_grad + opt_state / P);
            // ZeRO-2: also shard gradients
            double z2  = n * (bytes_param + bytes_grad / P + opt_state / P);
            // ZeRO-3: also shard parameters
            double z3  = n * (bytes_param + bytes_grad + opt_state) / P;
            printf("  %-8s %7.0f GB %7.1f GB %7.1f GB %7.1f GB\n",
                   mdl.name, ddp/1e9, z1/1e9, z2/1e9, z3/1e9);
        }
        printf("\n  ZeRO-3 divides ALL training memory by P. A 70B model that\n");
        printf("  needs 1120 GB/GPU under DDP fits in 140 GB/GPU under ZeRO-3\n");
        printf("  at P=8 -- and keeps shrinking as you add GPUs.\n");
    }

    // ── B: Simulate ZeRO-1 optimizer-state sharding ───────────────────────────
    section("B: ZeRO-1 simulation -- sharded optimizer matches unsharded");
    {
        const int P = 4;          // data-parallel GPUs
        const int NP = 8192;      // total parameters
        const int BLK = 256;
        const float lr = 1e-3f, b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
        const int steps = 5;

        // Each GPU has the full gradient (after AllReduce in DDP), but only
        // updates its own 1/P shard of the parameters using its shard of the
        // optimizer state. Then the updated shards are all-gathered.

        // Reference: unsharded Adam on all NP parameters
        float *d_p_ref, *d_g, *d_m_ref, *d_v_ref;
        CUDA_CHECK(cudaMalloc(&d_p_ref, NP * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_g,     NP * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_m_ref, NP * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_v_ref, NP * sizeof(float)));
        float* h_init = (float*)malloc(NP * sizeof(float));
        rand_fill(h_init, NP, -1.f, 1.f);
        CUDA_CHECK(cudaMemcpy(d_p_ref, h_init, NP*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_m_ref, 0, NP*sizeof(float)));
        CUDA_CHECK(cudaMemset(d_v_ref, 0, NP*sizeof(float)));

        // Sharded: same params/state but each shard updated independently
        float *d_p_shard, *d_m_shard, *d_v_shard;
        CUDA_CHECK(cudaMalloc(&d_p_shard, NP * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_m_shard, NP * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_v_shard, NP * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_p_shard, h_init, NP*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_m_shard, 0, NP*sizeof(float)));
        CUDA_CHECK(cudaMemset(d_v_shard, 0, NP*sizeof(float)));

        int shard = NP / P;

        for (int step = 1; step <= steps; step++) {
            // Fresh gradient each step (same for both paths)
            float* h_g = (float*)malloc(NP * sizeof(float));
            rand_fill(h_g, NP, -0.1f, 0.1f);
            CUDA_CHECK(cudaMemcpy(d_g, h_g, NP*sizeof(float), cudaMemcpyHostToDevice));
            free(h_g);

            float bc1 = 1.f - powf(b1, step);
            float bc2 = 1.f - powf(b2, step);

            // Reference: update all NP at once
            adamUpdateShard<<<(NP+BLK-1)/BLK, BLK>>>(
                d_p_ref, d_g, d_m_ref, d_v_ref, lr, b1, b2, eps, bc1, bc2, NP);

            // Sharded: each "GPU" updates only its shard, using its slice of
            // params/m/v and the corresponding slice of the gradient.
            for (int r = 0; r < P; r++) {
                int off = r * shard;
                adamUpdateShard<<<(shard+BLK-1)/BLK, BLK>>>(
                    d_p_shard + off, d_g + off,
                    d_m_shard + off, d_v_shard + off,
                    lr, b1, b2, eps, bc1, bc2, shard);
            }
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        float* h_ref   = (float*)malloc(NP * sizeof(float));
        float* h_shard = (float*)malloc(NP * sizeof(float));
        CUDA_CHECK(cudaMemcpy(h_ref,   d_p_ref,   NP*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_shard, d_p_shard, NP*sizeof(float), cudaMemcpyDeviceToHost));
        float err = max_abs_diff(h_ref, h_shard, NP);

        printf("  P=%d shards, %d params, %d Adam steps\n", P, NP, steps);
        printf("  Each shard owns %d params + their optimizer state\n", shard);
        printf("  Max diff (sharded vs unsharded): %.2e  %s\n", err,
               err < 1e-5f ? "PASS -- identical result, 1/P the state memory" : "FAIL");

        cudaFree(d_p_ref); cudaFree(d_g); cudaFree(d_m_ref); cudaFree(d_v_ref);
        cudaFree(d_p_shard); cudaFree(d_m_shard); cudaFree(d_v_shard);
        free(h_init); free(h_ref); free(h_shard);
    }

    // ── C: Memory vs communication trade-off ──────────────────────────────────
    section("C: ZeRO stage trade-offs");
    printf("  %-10s %-22s %-28s\n", "Stage", "Memory/GPU", "Extra communication");
    printf("  %-10s %-22s %-28s\n", "--------", "----------------------",
           "----------------------------");
    printf("  %-10s %-22s %-28s\n", "DDP",    "full replica",        "AllReduce gradients");
    printf("  %-10s %-22s %-28s\n", "ZeRO-1", "param+grad + opt/P",  "same as DDP");
    printf("  %-10s %-22s %-28s\n", "ZeRO-2", "param + (grad+opt)/P","same as DDP");
    printf("  %-10s %-22s %-28s\n", "ZeRO-3", "everything / P",      "+ all-gather params/layer");
    printf("\n  ZeRO-1 and ZeRO-2 are nearly free: same communication as DDP,\n");
    printf("  large memory savings. Use them by default for large models.\n");
    printf("  ZeRO-3 (FSDP) saves the most memory but all-gathers parameters\n");
    printf("  before every layer -- needs fast interconnect to stay efficient.\n\n");
    printf("  PyTorch: FullyShardedDataParallel(model) implements ZeRO-3.\n");
    printf("  DeepSpeed: ds_config {\"zero_optimization\": {\"stage\": 1|2|3}}.\n");

    return 0;
}
