// 03_nccl_collectives.cu  --  Post 8: Multi-GPU Infrastructure
//
// Demonstrates the NCCL (NVIDIA Collective Communications Library) API, which
// is what PyTorch, Megatron, and every production training stack actually use.
// NCCL implements ring and tree algorithms internally, auto-detects NVLink vs
// PCIe topology, and tunes the algorithm to the hardware.
//
// This program shows the four collectives that matter for distributed ML:
//   AllReduce      -- sum across GPUs, result on all (data-parallel gradients)
//   Broadcast      -- one GPU's data copied to all (model init, parameter sync)
//   AllGather      -- concat each GPU's shard onto all (tensor-parallel outputs)
//   ReduceScatter  -- sum then shard across GPUs (ZeRO optimizer, tensor-parallel)
//
// NCCL is OPTIONAL. If <nccl.h> is not installed, this file compiles to a stub
// that explains how to install it. If installed but only one GPU is present,
// NCCL still runs (single-GPU communicator) so you can verify the API calls.
//
// Build note: requires -lnccl when NCCL is available. The Makefile detects
// this automatically. To install NCCL:
//   - Ubuntu: apt install libnccl2 libnccl-dev
//   - conda:  conda install -c conda-forge nccl
//   - Or download from developer.nvidia.com/nccl

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "../include/utils.cuh"

#if __has_include(<nccl.h>)
#include <nccl.h>
#define HAVE_NCCL 1

#define NCCL_CHECK(call)                                                        \
    do {                                                                        \
        ncclResult_t r_ = (call);                                              \
        if (r_ != ncclSuccess) {                                               \
            fprintf(stderr, "NCCL error %s:%d  %s\n",                          \
                    __FILE__, __LINE__, ncclGetErrorString(r_));               \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// ─────────────────────────────────────────────────────────────────────────────
// Run all four collectives on a set of GPUs within one process.
// Uses ncclCommInitAll for single-process multi-GPU (the simplest NCCL setup).
// ─────────────────────────────────────────────────────────────────────────────
static void runNcclCollectives(int num_gpus) {
    const int N = 1 << 20;   // 1M elements per GPU

    // Create one communicator per GPU
    ncclComm_t* comms = (ncclComm_t*)malloc(num_gpus * sizeof(ncclComm_t));
    int* devs = (int*)malloc(num_gpus * sizeof(int));
    for (int i = 0; i < num_gpus; i++) devs[i] = i;
    NCCL_CHECK(ncclCommInitAll(comms, num_gpus, devs));

    // Per-GPU buffers and streams
    float**       send = (float**)malloc(num_gpus * sizeof(float*));
    float**       recv = (float**)malloc(num_gpus * sizeof(float*));
    cudaStream_t* streams = (cudaStream_t*)malloc(num_gpus * sizeof(cudaStream_t));

    for (int i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaMalloc(&send[i], (size_t)N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&recv[i], (size_t)N * num_gpus * sizeof(float)));
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
        // GPU i initialised with value (i+1)
        float* h = (float*)malloc((size_t)N * sizeof(float));
        for (int k = 0; k < N; k++) h[k] = (float)(i + 1);
        CUDA_CHECK(cudaMemcpy(send[i], h, (size_t)N * sizeof(float),
                              cudaMemcpyHostToDevice));
        free(h);
    }

    // ── AllReduce (sum) ───────────────────────────────────────────────────────
    section("NCCL AllReduce (sum)");
    NCCL_CHECK(ncclGroupStart());
    for (int i = 0; i < num_gpus; i++)
        NCCL_CHECK(ncclAllReduce(send[i], recv[i], N, ncclFloat, ncclSum,
                                 comms[i], streams[i]));
    NCCL_CHECK(ncclGroupEnd());
    for (int i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }
    {
        float expected = (float)(num_gpus * (num_gpus + 1) / 2);
        float h0; CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaMemcpy(&h0, recv[0], sizeof(float), cudaMemcpyDeviceToHost));
        printf("  Each GPU summed to: %.0f  (expected %.0f)  %s\n",
               h0, expected, fabsf(h0 - expected) < 1e-3f ? "PASS" : "FAIL");
        printf("  Use: averaging gradients in data-parallel training.\n");
    }

    // ── Broadcast (from GPU 0) ────────────────────────────────────────────────
    section("NCCL Broadcast (root = GPU 0)");
    NCCL_CHECK(ncclGroupStart());
    for (int i = 0; i < num_gpus; i++)
        NCCL_CHECK(ncclBroadcast(send[i], recv[i], N, ncclFloat, 0,
                                 comms[i], streams[i]));
    NCCL_CHECK(ncclGroupEnd());
    for (int i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }
    printf("  GPU 0's data (value 1) broadcast to all GPUs.\n");
    printf("  Use: distributing initial model weights to all workers.\n");

    // ── AllGather ─────────────────────────────────────────────────────────────
    section("NCCL AllGather");
    NCCL_CHECK(ncclGroupStart());
    for (int i = 0; i < num_gpus; i++)
        NCCL_CHECK(ncclAllGather(send[i], recv[i], N, ncclFloat,
                                 comms[i], streams[i]));
    NCCL_CHECK(ncclGroupEnd());
    for (int i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }
    printf("  Each GPU now holds all %d shards concatenated.\n", num_gpus);
    printf("  Use: gathering tensor-parallel partial outputs.\n");

    // ── ReduceScatter ─────────────────────────────────────────────────────────
    section("NCCL ReduceScatter");
    NCCL_CHECK(ncclGroupStart());
    for (int i = 0; i < num_gpus; i++)
        NCCL_CHECK(ncclReduceScatter(recv[i], send[i], N / num_gpus, ncclFloat,
                                     ncclSum, comms[i], streams[i]));
    NCCL_CHECK(ncclGroupEnd());
    for (int i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }
    printf("  Sum computed, then each GPU keeps only its shard of the result.\n");
    printf("  Use: ZeRO optimizer sharding, tensor-parallel reductions.\n");
    printf("  Note: AllReduce = ReduceScatter + AllGather (the ring algorithm).\n");

    // ── Bandwidth benchmark ───────────────────────────────────────────────────
    section("NCCL AllReduce bandwidth");
    {
        const int REPS = 50;
        CUDA_CHECK(cudaSetDevice(0));
        GpuTimer t; t.start(streams[0]);
        for (int r = 0; r < REPS; r++) {
            NCCL_CHECK(ncclGroupStart());
            for (int i = 0; i < num_gpus; i++)
                NCCL_CHECK(ncclAllReduce(send[i], recv[i], N, ncclFloat, ncclSum,
                                         comms[i], streams[i]));
            NCCL_CHECK(ncclGroupEnd());
        }
        for (int i = 0; i < num_gpus; i++) {
            CUDA_CHECK(cudaSetDevice(i));
            CUDA_CHECK(cudaStreamSynchronize(streams[i]));
        }
        float ms = t.stop_ms(streams[0]) / REPS;
        // Algorithm bandwidth: bytes processed / time
        double data_bytes = (double)N * sizeof(float);
        // Bus bandwidth (the NCCL convention): 2*(P-1)/P * data
        double bus_bytes = 2.0 * (num_gpus - 1) / num_gpus * data_bytes;
        printf("  Message size : %.1f MB\n", data_bytes / 1e6);
        printf("  Time/AllReduce: %.3f ms\n", ms);
        printf("  Bus bandwidth : %.1f GB/s\n", bw_gb_s((size_t)bus_bytes, ms));
        printf("  (Bus BW is the NCCL-reported metric: 2*(P-1)/P * size / time)\n");
    }

    // Cleanup
    for (int i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        cudaFree(send[i]); cudaFree(recv[i]);
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
        ncclCommDestroy(comms[i]);
    }
    free(send); free(recv); free(streams); free(comms); free(devs);
}

int main() {
    print_device_info();
    int num_gpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));

    printf("NCCL is available. Running collectives on %d GPU%s.\n",
           num_gpus, num_gpus == 1 ? "" : "s");
    if (num_gpus == 1)
        printf("(Single-GPU communicator: API works, but no real cross-GPU traffic.)\n");

    runNcclCollectives(num_gpus);

    section("How PyTorch uses NCCL");
    printf("  torch.distributed.init_process_group(backend='nccl')\n");
    printf("  DistributedDataParallel wraps your model and calls ncclAllReduce\n");
    printf("  on gradients automatically during loss.backward().\n");
    printf("  The collectives above are exactly what runs under the hood.\n");
    return 0;
}

#else  // ───────────────────────────────────────────────────────────────────────
// NCCL not installed: compile a stub that explains how to get it.

int main() {
    print_device_info();
    section("NCCL not found");
    printf("  This program demonstrates NCCL collectives but <nccl.h> was not\n");
    printf("  found at compile time, so it was built as a stub.\n\n");
    printf("  To install NCCL:\n");
    printf("    Ubuntu/Debian : sudo apt install libnccl2 libnccl-dev\n");
    printf("    conda         : conda install -c conda-forge nccl\n");
    printf("    Manual        : developer.nvidia.com/nccl\n\n");
    printf("  Then rebuild:  make 03_nccl_collectives\n\n");
    printf("  In the meantime, program 02 (ring-AllReduce from scratch) shows\n");
    printf("  the exact algorithm NCCL implements internally -- and it runs\n");
    printf("  without NCCL on any number of GPUs.\n");
    return 0;
}

#endif
