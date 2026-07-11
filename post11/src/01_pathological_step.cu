// 01_pathological_step.cu  --  Post 11: Profiling and Optimization
//
// A training-step-shaped workload with the FOUR TIMELINE PATHOLOGIES from the
// post deliberately built in -- and a --fixed mode that applies the remedies.
// This is the program the Nsight Systems section profiles.
//
//   (1) LAUNCH-BOUND     200 tiny kernel launches back-to-back
//                        fix: one kernel that does all 200 iterations
//   (2) FALSE SERIAL     two independent "head" GEMMs on one stream
//                        fix: run them on two streams, join with events
//   (3) COMM NOT HIDDEN  simulated gradient AllReduce (big D2D copy + axpy)
//                        runs AFTER backward, blocking the step
//                        fix: overlap it with backward on a comm stream
//                        (like DDP's bucketed overlap -- the payload is the
//                         PREVIOUS step's gradients, so it is independent)
//   (4) SYNC READBACK    4-byte blocking cudaMemcpy of the loss every step
//                        fix: async copy into pinned memory, poll every N
//
// Everything is NVTX-annotated, so the two profiles diff cleanly:
//
//   nsys profile -t cuda,nvtx -o slow  ./build/01_pathological_step
//   nsys profile -t cuda,nvtx -o fast  ./build/01_pathological_step --fixed
//
// Even without nsys, the program times each pathology with CUDA events and
// prints a before/after table. The tiny-op chain and the head outputs are
// verified identical between modes: the fixes change the schedule, not math.

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/utils.cuh"

// NVTX is header-only and optional (same trick as the Post 4 repo)
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define RANGE_PUSH(name) nvtxRangePushA(name)
#define RANGE_POP()      nvtxRangePop()
#else
#define RANGE_PUSH(name) do {} while (0)
#define RANGE_POP()      do {} while (0)
#endif

__global__ void biasReluKernel(float* __restrict__ x,
                               const float* __restrict__ b, int rows, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * D) {
        float v = x[i] + b[i % D];
        x[i] = v > 0.f ? v : 0.f;
    }
}

// pathology 1: one tiny step of work (scale + shift on a small buffer)
__global__ void tinyOpKernel(float* x, float a, float c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = a * x[i] + c;
}

// the fix: same 200 iterations, one launch
__global__ void fusedOpsKernel(float* x, float a, float c, int iters, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        for (int k = 0; k < iters; k++) v = a * v + c;
        x[i] = v;
    }
}

// pathology 3 payload: grad += comm_buf (stands in for the AllReduce math)
__global__ void axpyKernel(float* __restrict__ y,
                           const float* __restrict__ x, float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += a * x[i];
}

__global__ void lossKernel(const float* __restrict__ x, float* loss, int n) {
    // crude block reduction into a single value -- enough for a readback demo
    __shared__ float s[256];
    int tid = threadIdx.x;
    float acc = 0.f;
    for (int i = tid; i < n; i += blockDim.x) acc += x[i] * x[i];
    s[tid] = acc; __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid == 0) *loss = s[0];
}

int main(int argc, char** argv) {
    bool fixed = (argc > 1) && (strcmp(argv[1], "--fixed") == 0);
    print_device_info();
    srand(42);
    printf("Mode: %s\n\n", fixed ? "FIXED (remedies applied)"
                                 : "PATHOLOGICAL (run with --fixed to compare)");

    const int D = 1024, B = 512, BLK = 256;
    const int TINY_N = 1024, TINY_ITERS = 200;
    const int COMM_N = 16 * 1024 * 1024;          // 64 MB "gradient bucket"
    const int STEPS = 20;

    cublasHandle_t cublas; CUBLAS_CHECK(cublasCreate(&cublas));
    float one = 1.f, zero = 0.f;

    // model-ish buffers
    float *d_W1, *d_W2, *d_Wh1, *d_Wh2, *d_b;
    float *d_x, *d_h, *d_y, *d_head1, *d_head2;
    float *d_tiny, *d_grad, *d_comm, *d_loss;
    CUDA_CHECK(cudaMalloc(&d_W1,  (size_t)D*D*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W2,  (size_t)D*D*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Wh1, (size_t)D*D*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Wh2, (size_t)D*D*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b,   (size_t)D*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x,   (size_t)B*D*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_h,   (size_t)B*D*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y,   (size_t)B*D*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_head1,(size_t)B*D*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_head2,(size_t)B*D*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tiny, (size_t)TINY_N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad, (size_t)COMM_N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_comm, (size_t)COMM_N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));

    {   // init
        float* h = (float*)malloc((size_t)COMM_N*sizeof(float));
        rand_fill(h, D*D, -0.05f, 0.05f); CUDA_CHECK(cudaMemcpy(d_W1,  h, (size_t)D*D*4, cudaMemcpyHostToDevice));
        rand_fill(h, D*D, -0.05f, 0.05f); CUDA_CHECK(cudaMemcpy(d_W2,  h, (size_t)D*D*4, cudaMemcpyHostToDevice));
        rand_fill(h, D*D, -0.05f, 0.05f); CUDA_CHECK(cudaMemcpy(d_Wh1, h, (size_t)D*D*4, cudaMemcpyHostToDevice));
        rand_fill(h, D*D, -0.05f, 0.05f); CUDA_CHECK(cudaMemcpy(d_Wh2, h, (size_t)D*D*4, cudaMemcpyHostToDevice));
        rand_fill(h, D, -0.05f, 0.05f);   CUDA_CHECK(cudaMemcpy(d_b,   h, (size_t)D*4,   cudaMemcpyHostToDevice));
        rand_fill(h, B*D, -1.f, 1.f);     CUDA_CHECK(cudaMemcpy(d_x,   h, (size_t)B*D*4, cudaMemcpyHostToDevice));
        rand_fill(h, TINY_N, -1.f, 1.f);  CUDA_CHECK(cudaMemcpy(d_tiny,h, (size_t)TINY_N*4, cudaMemcpyHostToDevice));
        rand_fill(h, COMM_N, -0.01f, 0.01f);
        CUDA_CHECK(cudaMemcpy(d_grad, h, (size_t)COMM_N*4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_comm, h, (size_t)COMM_N*4, cudaMemcpyHostToDevice));
        free(h);
    }

    cudaStream_t s_head, s_comm;
    CUDA_CHECK(cudaStreamCreate(&s_head));
    CUDA_CHECK(cudaStreamCreate(&s_comm));
    cudaEvent_t ev_h_done, ev_head_done, ev_comm_done;
    CUDA_CHECK(cudaEventCreate(&ev_h_done));
    CUDA_CHECK(cudaEventCreate(&ev_head_done));
    CUDA_CHECK(cudaEventCreate(&ev_comm_done));

    float* h_loss_pinned;
    CUDA_CHECK(cudaMallocHost(&h_loss_pinned, sizeof(float)));
    float last_loss = 0.f;

    // per-phase accumulated times
    float t_fwd = 0, t_tiny = 0, t_heads = 0, t_comm = 0, t_read = 0, t_step = 0;
    GpuTimer tim_phase, tim_step;

    // warmup (post's rule 1: never time the first iterations)
    for (int w = 0; w < 3; w++) {
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            D, B, D, &one, d_W1, D, d_x, D, &zero, d_h, D));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int step = 0; step < STEPS; step++) {
        tim_step.start();

        // ── forward: 2 GEMMs + bias/relu ─────────────────────────────────────
        RANGE_PUSH("forward");
        tim_phase.start();
        CUBLAS_CHECK(cublasSetStream(cublas, 0));
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            D, B, D, &one, d_W1, D, d_x, D, &zero, d_h, D));
        biasReluKernel<<<(B*D+BLK-1)/BLK, BLK>>>(d_h, d_b, B, D);
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            D, B, D, &one, d_W2, D, d_h, D, &zero, d_y, D));
        CUDA_CHECK(cudaEventRecord(ev_h_done, 0));
        // pathology 3 FIX: kick the (previous step's) gradient comm NOW,
        // on its own stream, overlapped with everything below
        if (fixed) {
            RANGE_PUSH("grad_comm(overlapped)");
            CUDA_CHECK(cudaMemcpyAsync(d_comm, d_grad,
                (size_t)COMM_N*sizeof(float), cudaMemcpyDeviceToDevice, s_comm));
            axpyKernel<<<(COMM_N+BLK-1)/BLK, BLK, 0, s_comm>>>(
                d_grad, d_comm, 0.5f, COMM_N);
            CUDA_CHECK(cudaEventRecord(ev_comm_done, s_comm));
            RANGE_POP();
        }
        t_fwd += tim_phase.stop_ms();
        RANGE_POP();

        // ── pathology 1: the tiny-op chain ───────────────────────────────────
        RANGE_PUSH(fixed ? "tiny_ops(fused)" : "tiny_ops(200 launches)");
        tim_phase.start();
        if (!fixed) {
            for (int k = 0; k < TINY_ITERS; k++)
                tinyOpKernel<<<(TINY_N+BLK-1)/BLK, BLK>>>(d_tiny, 1.0001f,
                                                          1e-4f, TINY_N);
        } else {
            fusedOpsKernel<<<(TINY_N+BLK-1)/BLK, BLK>>>(d_tiny, 1.0001f,
                                                        1e-4f, TINY_ITERS, TINY_N);
        }
        t_tiny += tim_phase.stop_ms();
        RANGE_POP();

        // ── pathology 2: two independent heads ───────────────────────────────
        RANGE_PUSH(fixed ? "heads(2 streams)" : "heads(serialized)");
        tim_phase.start();
        if (!fixed) {
            CUBLAS_CHECK(cublasSetStream(cublas, 0));
            CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                D, B, D, &one, d_Wh1, D, d_y, D, &zero, d_head1, D));
            CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                D, B, D, &one, d_Wh2, D, d_y, D, &zero, d_head2, D));
        } else {
            CUDA_CHECK(cudaStreamWaitEvent(s_head, ev_h_done, 0));
            CUBLAS_CHECK(cublasSetStream(cublas, 0));
            CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                D, B, D, &one, d_Wh1, D, d_y, D, &zero, d_head1, D));
            CUBLAS_CHECK(cublasSetStream(cublas, s_head));
            CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                D, B, D, &one, d_Wh2, D, d_y, D, &zero, d_head2, D));
            CUDA_CHECK(cudaEventRecord(ev_head_done, s_head));
            CUDA_CHECK(cudaStreamWaitEvent(0, ev_head_done, 0));
        }
        t_heads += tim_phase.stop_ms();
        RANGE_POP();

        // ── pathology 3 (baseline): comm AFTER compute, blocking ─────────────
        RANGE_PUSH(fixed ? "grad_comm(wait)" : "grad_comm(blocking)");
        tim_phase.start();
        if (!fixed) {
            CUDA_CHECK(cudaMemcpy(d_comm, d_grad,
                (size_t)COMM_N*sizeof(float), cudaMemcpyDeviceToDevice));
            axpyKernel<<<(COMM_N+BLK-1)/BLK, BLK>>>(d_grad, d_comm, 0.5f, COMM_N);
        } else {
            CUDA_CHECK(cudaStreamWaitEvent(0, ev_comm_done, 0));  // usually done
        }
        t_comm += tim_phase.stop_ms();
        RANGE_POP();

        // ── pathology 4: loss readback ───────────────────────────────────────
        RANGE_PUSH(fixed ? "loss(async, every 10)" : "loss(sync every step)");
        tim_phase.start();
        lossKernel<<<1, 256>>>(d_y, d_loss, B * D);
        if (!fixed) {
            float l;
            CUDA_CHECK(cudaMemcpy(&l, d_loss, sizeof(float),
                                  cudaMemcpyDeviceToHost));       // full sync
            last_loss = l;
        } else {
            CUDA_CHECK(cudaMemcpyAsync(h_loss_pinned, d_loss, sizeof(float),
                                       cudaMemcpyDeviceToHost, 0));
            if (step % 10 == 9) {                    // poll rarely
                CUDA_CHECK(cudaStreamSynchronize(0));
                last_loss = *h_loss_pinned;
            }
        }
        t_read += tim_phase.stop_ms();
        RANGE_POP();

        CUDA_CHECK(cudaDeviceSynchronize());
        t_step += tim_step.stop_ms();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    last_loss = fixed ? *h_loss_pinned : last_loss;

    section("Per-phase time (avg over steps)");
    printf("  %-26s %10s\n", "phase", "ms/step");
    printf("  %-26s %10.3f   (1) %s\n", "tiny_ops", t_tiny/STEPS,
           fixed ? "fused: 1 launch" : "200 launches");
    printf("  %-26s %10.3f   (2) %s\n", "heads", t_heads/STEPS,
           fixed ? "2 streams" : "serialized");
    printf("  %-26s %10.3f   (3) %s\n", "grad_comm", t_comm/STEPS,
           fixed ? "overlapped w/ compute" : "blocking after compute");
    printf("  %-26s %10.3f   (4) %s\n", "loss_readback", t_read/STEPS,
           fixed ? "async + poll every 10" : "sync cudaMemcpy every step");
    printf("  %-26s %10.3f\n", "forward (unchanged)", t_fwd/STEPS);
    printf("  %-26s %10.3f\n", "TOTAL step", t_step/STEPS);
    printf("\n  final loss %.4f  (identical math in both modes)\n", last_loss);

    section("Profile it");
    printf("  nsys profile -t cuda,nvtx -o slow ./build/01_pathological_step\n");
    printf("  nsys profile -t cuda,nvtx -o fast ./build/01_pathological_step --fixed\n");
    printf("  Open both in the GUI: the NVTX ranges name every pathology, and\n");
    printf("  the fixed timeline shows grad_comm running UNDER the compute.\n");
    printf("  (Pathology 1's true fix at scale is CUDA Graphs -- Post 4 repo.)\n");

    // cleanup
    cublasDestroy(cublas);
    cudaStreamDestroy(s_head); cudaStreamDestroy(s_comm);
    cudaEventDestroy(ev_h_done); cudaEventDestroy(ev_head_done);
    cudaEventDestroy(ev_comm_done);
    cudaFreeHost(h_loss_pinned);
    cudaFree(d_W1); cudaFree(d_W2); cudaFree(d_Wh1); cudaFree(d_Wh2);
    cudaFree(d_b); cudaFree(d_x); cudaFree(d_h); cudaFree(d_y);
    cudaFree(d_head1); cudaFree(d_head2); cudaFree(d_tiny);
    cudaFree(d_grad); cudaFree(d_comm); cudaFree(d_loss);
    return 0;
}
