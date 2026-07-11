// 02_pipeline_parallel.cu  --  Post 10: Pipeline Parallelism and MoE
//
// Runs a REAL pipelined forward pass on one GPU, using CUDA streams as
// virtual pipeline stages and events as the stage-to-stage handoffs.
//
// Model: an 8-layer MLP (each layer: X @ W + bias + ReLU), split into
// P=4 stages of 2 layers each. The batch is split into M micro-batches.
//
//   stage k, micro-batch i:
//     cudaStreamWaitEvent(stream[k], done[k-1][i])   <- wait for upstream
//     ... run this stage's layers on micro-batch i ...
//     cudaEventRecord(done[k][i], stream[k])         <- release downstream
//
// This is exactly the dependency structure a multi-GPU pipeline framework
// builds -- there, the event/wait pair becomes an NCCL send/recv between
// GPUs, but the schedule is the same. On one GPU the stages share SMs, so
// don't expect a big speedup; what this program demonstrates is:
//
//   1. CORRECTNESS: the pipelined result is bit-identical in structure to
//      the sequential full-batch reference (verified).
//   2. THE SCHEDULE IS REAL: per-op timestamps are recorded with events and
//      printed as a measured Gantt chart -- you can see stage 1 working on
//      micro-batch 0 while stage 0 is already on micro-batch 1.

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include "../include/utils.cuh"

// bias + ReLU, applied to a [rows x D] slice
__global__ void biasReluKernel(float* __restrict__ x,
                               const float* __restrict__ b,
                               int rows, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * D) {
        int col = i % D;
        float v = x[i] + b[col];
        x[i] = v > 0.f ? v : 0.f;
    }
}

int main(int argc, char** argv) {
    print_device_info();
    srand(42);

    const int D      = 1024;   // width (all layers D -> D)
    const int B      = 1024;   // global batch (rows)
    const int LAYERS = 8;
    const int P      = 4;      // pipeline stages
    const int M      = (argc > 1) ? atoi(argv[1]) : 4;   // micro-batches
    const int LPS    = LAYERS / P;   // layers per stage
    const int mb     = B / M;        // rows per micro-batch
    const int BLK    = 256;

    if (M < 1 || M > 16 || B % M != 0) {
        fprintf(stderr, "M must be 1..16 and divide B=%d (got %d)\n", B, M);
        return 1;
    }

    printf("Pipelined MLP: %d layers of [%d x %d], split into P=%d stages\n",
           LAYERS, D, D, P);
    printf("Batch %d split into M=%d micro-batches of %d rows\n\n", B, M, mb);

    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));
    float one = 1.f, zero = 0.f;

    // ── Weights and biases per layer ──────────────────────────────────────────
    float* d_W[LAYERS]; float* d_b[LAYERS];
    float* h_tmp = (float*)malloc((size_t)D * D * sizeof(float));
    for (int l = 0; l < LAYERS; l++) {
        CUDA_CHECK(cudaMalloc(&d_W[l], (size_t)D * D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_b[l], (size_t)D * sizeof(float)));
        rand_fill(h_tmp, D * D, -0.05f, 0.05f);
        CUDA_CHECK(cudaMemcpy(d_W[l], h_tmp, (size_t)D*D*sizeof(float),
                              cudaMemcpyHostToDevice));
        rand_fill(h_tmp, D, -0.05f, 0.05f);
        CUDA_CHECK(cudaMemcpy(d_b[l], h_tmp, (size_t)D*sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    // ── Input and stage-boundary activation buffers ──────────────────────────
    // act[k] = activations entering stage k (act[P] = final output).
    // Micro-batch i occupies rows [i*mb, (i+1)*mb) -- disjoint slices, so all
    // micro-batches can be in flight at once without aliasing.
    float* d_act[P + 1];
    for (int k = 0; k <= P; k++)
        CUDA_CHECK(cudaMalloc(&d_act[k], (size_t)B * D * sizeof(float)));
    // stage-internal scratch (between the two layers of a stage), per stage
    float* d_mid[P];
    for (int k = 0; k < P; k++)
        CUDA_CHECK(cudaMalloc(&d_mid[k], (size_t)B * D * sizeof(float)));

    float* h_in = (float*)malloc((size_t)B * D * sizeof(float));
    rand_fill(h_in, B * D, -1.f, 1.f);
    CUDA_CHECK(cudaMemcpy(d_act[0], h_in, (size_t)B*D*sizeof(float),
                          cudaMemcpyHostToDevice));

    // ── One stage's compute for one micro-batch, on a given stream ────────────
    // in  -> W[first] -> mid -> W[first+1] -> out       (LPS = 2)
    // (typed-parameter lambda: fine in C++17; only auto-parameter lambdas
    //  are C++20 and banned by our build rules)
    auto run_stage = [&](int k, int i, float* in, float* out, cudaStream_t st) {
        CUBLAS_CHECK(cublasSetStream(cublas, st));
        float* src = in  + (size_t)i * mb * D;
        float* tmp = d_mid[k] + (size_t)i * mb * D;
        float* dst = out + (size_t)i * mb * D;
        int l0 = k * LPS;
        // layer l0: tmp = src @ W
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            D, mb, D, &one, d_W[l0], D, src, D, &zero, tmp, D));
        biasReluKernel<<<(mb*D + BLK-1)/BLK, BLK, 0, st>>>(tmp, d_b[l0], mb, D);
        // layer l0+1: dst = tmp @ W
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            D, mb, D, &one, d_W[l0+1], D, tmp, D, &zero, dst, D));
        biasReluKernel<<<(mb*D + BLK-1)/BLK, BLK, 0, st>>>(dst, d_b[l0+1], mb, D);
    };

    // ── Reference: sequential full batch on the default stream ───────────────
    section("Reference: sequential full-batch forward");
    float* d_ref[2];
    CUDA_CHECK(cudaMalloc(&d_ref[0], (size_t)B * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ref[1], (size_t)B * D * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_ref[0], d_act[0], (size_t)B*D*sizeof(float),
                          cudaMemcpyDeviceToDevice));
    CUBLAS_CHECK(cublasSetStream(cublas, 0));
    GpuTimer t_ref; t_ref.start();
    {
        float *cur = d_ref[0], *nxt = d_ref[1];
        for (int l = 0; l < LAYERS; l++) {
            CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                D, B, D, &one, d_W[l], D, cur, D, &zero, nxt, D));
            biasReluKernel<<<(B*D + BLK-1)/BLK, BLK>>>(nxt, d_b[l], B, D);
            float* sw = cur; cur = nxt; nxt = sw;
        }
        // after 8 layers (even), result is back in d_ref[0]
    }
    float ms_ref = t_ref.stop_ms();
    printf("  8 layers, full batch, one stream: %.3f ms\n", ms_ref);

    // ── Pipelined: P streams, M micro-batches, event handoffs ────────────────
    section("Pipelined: P streams as virtual stages");
    cudaStream_t stream[P];
    cudaEvent_t  done[P][16];               // done[k][i], M <= 16
    cudaEvent_t  op_beg[P][16], op_end[P][16];
    for (int k = 0; k < P; k++) {
        CUDA_CHECK(cudaStreamCreate(&stream[k]));
        for (int i = 0; i < M; i++) {
            CUDA_CHECK(cudaEventCreate(&done[k][i]));
            CUDA_CHECK(cudaEventCreate(&op_beg[k][i]));
            CUDA_CHECK(cudaEventCreate(&op_end[k][i]));
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t t0; CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventRecord(t0, 0));
    GpuTimer t_pipe; t_pipe.start();

    // launch order: micro-batch outer, stage inner (any order works --
    // the events enforce the true dependencies)
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < P; k++) {
            if (k > 0)
                CUDA_CHECK(cudaStreamWaitEvent(stream[k], done[k-1][i], 0));
            CUDA_CHECK(cudaEventRecord(op_beg[k][i], stream[k]));
            run_stage(k, i, d_act[k], d_act[k+1], stream[k]);
            CUDA_CHECK(cudaEventRecord(op_end[k][i], stream[k]));
            CUDA_CHECK(cudaEventRecord(done[k][i], stream[k]));
        }
    }
    for (int k = 0; k < P; k++) CUDA_CHECK(cudaStreamSynchronize(stream[k]));
    float ms_pipe = t_pipe.stop_ms();
    printf("  P=%d streams, M=%d micro-batches:   %.3f ms\n", P, M, ms_pipe);
    printf("  (one GPU: stages share SMs, so ~equal time is expected;\n");
    printf("   on P GPUs the stages run on separate silicon)\n");

    // ── Verify ────────────────────────────────────────────────────────────────
    section("Verification");
    float* h_ref = (float*)malloc((size_t)B * D * sizeof(float));
    float* h_out = (float*)malloc((size_t)B * D * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_ref, d_ref[0],  (size_t)B*D*sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_out, d_act[P],  (size_t)B*D*sizeof(float),
                          cudaMemcpyDeviceToHost));
    float err = max_abs_diff(h_ref, h_out, B * D);
    printf("  max |pipelined - sequential| = %.2e  %s\n", err,
           err < 1e-4f ? "PASS -- pipelining changes the schedule, not the math"
                       : "FAIL");

    // ── Measured Gantt: when did each (stage, micro-batch) actually run? ─────
    section("Measured schedule (ms after start)");
    printf("  %-8s", "");
    for (int i = 0; i < M; i++) printf("      mb%-6d", i);
    printf("\n");
    for (int k = 0; k < P; k++) {
        printf("  stage %d ", k);
        for (int i = 0; i < M; i++) {
            float b_ms, e_ms;
            CUDA_CHECK(cudaEventElapsedTime(&b_ms, t0, op_beg[k][i]));
            CUDA_CHECK(cudaEventElapsedTime(&e_ms, t0, op_end[k][i]));
            printf(" [%5.2f-%5.2f]", b_ms, e_ms);
        }
        printf("\n");
    }
    printf("\n  Read down a column: micro-batch i marches through the stages.\n");
    printf("  Read across overlapping intervals: while stage 1 works on mb0,\n");
    printf("  stage 0 has already started mb1 -- that overlap IS the pipeline.\n");
    printf("  The empty staircase before stage 3's first op is the fill bubble\n");
    printf("  from program 01, now measured on real hardware.\n");

    // Cleanup
    for (int k = 0; k < P; k++) {
        CUDA_CHECK(cudaStreamDestroy(stream[k]));
        for (int i = 0; i < M; i++) {
            cudaEventDestroy(done[k][i]);
            cudaEventDestroy(op_beg[k][i]); cudaEventDestroy(op_end[k][i]);
        }
        cudaFree(d_mid[k]);
    }
    cudaEventDestroy(t0);
    for (int k = 0; k <= P; k++) cudaFree(d_act[k]);
    for (int l = 0; l < LAYERS; l++) { cudaFree(d_W[l]); cudaFree(d_b[l]); }
    cudaFree(d_ref[0]); cudaFree(d_ref[1]);
    cublasDestroy(cublas);
    free(h_tmp); free(h_in); free(h_ref); free(h_out);
    return 0;
}
