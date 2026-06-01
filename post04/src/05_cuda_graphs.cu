// 05_cuda_graphs.cu — Post 4: CUDA Streams and Async Execution
//
// Every kernel launch has a CPU-side overhead of roughly 5–20 µs:
// packaging arguments, acquiring driver locks, writing the command to the
// GPU's command queue.  For large kernels with long runtimes this is noise.
// For tight loops with many small kernels it accumulates and can become the
// dominant cost.
//
// CUDA Graphs solve this by recording a sequence of operations ONCE and
// replaying the entire sequence as a single GPU-side submission.  After the
// graph is instantiated, each replay:
//   - Issues one GPU command instead of N
//   - Requires no CPU involvement during replay (beyond the Launch call)
//   - Achieves near-zero per-kernel CPU overhead
//
// When to use CUDA Graphs:
//   ✓ Inference loops with fixed input shapes
//   ✓ Training loops with static batch size and model (torch.compile uses this)
//   ✓ Any tight loop where nsys shows visible gaps between small kernels
//
// When NOT to use:
//   ✗ Dynamic shapes (different sizes each step)
//   ✗ Loops with conditional logic that changes between iterations
//   ✗ When the per-kernel time already dwarfs launch overhead
//
// This program demonstrates:
//   A. Measuring raw launch overhead (many tiny kernels back-to-back)
//   B. Capturing a multi-kernel graph and replaying it
//   C. Updating graph arguments (for changing inputs between replays)

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/timer.cuh"

// ─────────────────────────────────────────────────────────────────────────────
// A deliberately SMALL kernel so launch overhead dominates runtime.
// This is the regime where CUDA Graphs help the most.
// ─────────────────────────────────────────────────────────────────────────────

__global__ void tinyKernel(float* data, float scalar, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = data[i] * scalar + (1.0f - scalar);
}

// A heavier kernel to show that graphs still help even when kernels are larger
__global__ void heavyKernel(float* data, float scalar, int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = data[i];
    for (int it = 0; it < iters; it++) v = v * scalar + 0.0001f;
    data[i] = v;
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    print_device_info();

    // Small array: kernel finishes fast, so launch overhead is a large fraction
    const int N_SMALL   = 1 << 14;   // 16 K floats — kernel runs in ~1–5 µs
    const int N_LARGE   = 1 << 22;   // 4 M floats  — kernel runs in ~1–5 ms
    const int NUM_STEPS = 100;        // simulate 100 training/inference steps
    const int BLOCK     = 256;
    const int GRID_S    = (N_SMALL + BLOCK - 1) / BLOCK;
    const int GRID_L    = (N_LARGE + BLOCK - 1) / BLOCK;

    float *d_s, *d_l;
    CUDA_CHECK(cudaMalloc(&d_s, N_SMALL * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_l, N_LARGE * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_s, 1, N_SMALL * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_l, 1, N_LARGE * sizeof(float)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // ═══════════════════════════════════════════════════════════════════════
    // A: Launch overhead — many tiny kernels, eager mode
    // ═══════════════════════════════════════════════════════════════════════
    section("A: Tiny kernels — eager launches vs CUDA Graph replay");

    // Measure one kernel time
    {
        tinyKernel<<<GRID_S, BLOCK, 0, stream>>>(d_s, 1.0001f, N_SMALL);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        GpuTimer t; t.start(stream);
        tinyKernel<<<GRID_S, BLOCK, 0, stream>>>(d_s, 1.0001f, N_SMALL);
        float ms_one = t.stop_ms(stream);
        printf("  One tinyKernel : %.4f ms  (%.0f µs)\n", ms_one, ms_one * 1000);
    }

    // Eager: launch STEPS kernels individually, measure total
    {
        tinyKernel<<<GRID_S, BLOCK, 0, stream>>>(d_s, 1.f, N_SMALL); // warm up
        CUDA_CHECK(cudaStreamSynchronize(stream));

        GpuTimer t; t.start(stream);
        for (int step = 0; step < NUM_STEPS; step++)
            tinyKernel<<<GRID_S, BLOCK, 0, stream>>>(d_s, 1.0001f, N_SMALL);
        float ms_eager = t.stop_ms(stream);

        float overhead_us = (ms_eager * 1000.0f) / NUM_STEPS;
        printf("  Eager %d launches : %.2f ms total  (%.1f µs/launch overhead)\n",
               NUM_STEPS, ms_eager, overhead_us);
    }

    // CUDA Graph: capture once, replay STEPS times
    {
        // ── Capture phase ─────────────────────────────────────────────────
        // Between BeginCapture and EndCapture, no GPU work is submitted.
        // Instead, operations are recorded into the graph structure.
        cudaGraph_t     graph;
        cudaGraphExec_t instance;

        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

        // Everything launched on 'stream' between Begin and End is captured.
        // You can have multiple kernels, memsets, and even stream-level events.
        tinyKernel<<<GRID_S, BLOCK, 0, stream>>>(d_s, 1.0001f, N_SMALL);

        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));

        // ── Instantiation ─────────────────────────────────────────────────
        // Compiles the graph into an executable form.  This is the slow step —
        // do it once outside the hot path.
        CUDA_CHECK(cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0));

        // Warm up the replay
        CUDA_CHECK(cudaGraphLaunch(instance, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // ── Replay phase ──────────────────────────────────────────────────
        GpuTimer t; t.start(stream);
        for (int step = 0; step < NUM_STEPS; step++)
            CUDA_CHECK(cudaGraphLaunch(instance, stream));
        float ms_graph = t.stop_ms(stream);

        float overhead_us = (ms_graph * 1000.0f) / NUM_STEPS;
        printf("  Graph  %d replays : %.2f ms total  (%.1f µs/replay overhead)\n",
               NUM_STEPS, ms_graph, overhead_us);

        CUDA_CHECK(cudaGraphExecDestroy(instance));
        CUDA_CHECK(cudaGraphDestroy(graph));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // B: Multi-kernel graph (simulating a training step)
    // ═══════════════════════════════════════════════════════════════════════
    section("B: Multi-kernel graph — simulating a fixed-shape training step");
    {
        // A simple "training step": multiple kernels in sequence.
        // In real code this would be forward + backward + optimizer.
        const int K = 8;   // number of kernels in the step

        // Eager baseline
        {
            for (int k = 0; k < K; k++)
                heavyKernel<<<GRID_L, BLOCK, 0, stream>>>(d_l, 1.0001f, N_LARGE, 20);
            CUDA_CHECK(cudaStreamSynchronize(stream));  // warm up

            GpuTimer t; t.start(stream);
            for (int step = 0; step < NUM_STEPS; step++)
                for (int k = 0; k < K; k++)
                    heavyKernel<<<GRID_L, BLOCK, 0, stream>>>(d_l, 1.0001f, N_LARGE, 20);
            float ms_eager = t.stop_ms(stream);
            printf("  Eager  %d steps × %d kernels : %.2f ms  (%.2f ms/step)\n",
                   NUM_STEPS, K, ms_eager, ms_eager / NUM_STEPS);
        }

        // Graph: capture the entire step, replay NUM_STEPS times
        {
            cudaGraph_t     graph;
            cudaGraphExec_t instance;

            CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
            for (int k = 0; k < K; k++)
                heavyKernel<<<GRID_L, BLOCK, 0, stream>>>(d_l, 1.0001f, N_LARGE, 20);
            CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
            CUDA_CHECK(cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0));

            // Warm up
            CUDA_CHECK(cudaGraphLaunch(instance, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            GpuTimer t; t.start(stream);
            for (int step = 0; step < NUM_STEPS; step++)
                CUDA_CHECK(cudaGraphLaunch(instance, stream));
            float ms_graph = t.stop_ms(stream);
            printf("  Graph  %d steps × %d kernels : %.2f ms  (%.2f ms/step)\n",
                   NUM_STEPS, K, ms_graph, ms_graph / NUM_STEPS);
            printf("  Speedup : %.2f×\n", ms_graph > 0 ? ms_graph/ms_graph : 1.f);
            // Note: for large kernels the speedup is small (launch overhead << kernel time)
            // For tiny kernels (part A) the speedup is dramatic.

            CUDA_CHECK(cudaGraphExecDestroy(instance));
            CUDA_CHECK(cudaGraphDestroy(graph));
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // C: Updating graph node parameters between replays
    // ═══════════════════════════════════════════════════════════════════════
    section("C: Updating kernel parameters between graph replays");
    {
        // A common need: replay the same graph structure but with different
        // data each time (e.g. different input tensor contents, same shape).
        //
        // Approach: the graph's kernel nodes store copies of the arguments.
        // cudaGraphExecKernelNodeSetParams() updates those stored arguments
        // without re-capturing the graph.
        //
        // Important: the SHAPE (grid, block, shared mem) cannot change.
        // Only scalar arguments and pointer values can be updated.

        float scalar_values[] = {1.0001f, 1.0002f, 0.9999f, 1.0003f};
        const int NUM_SCALARS = 4;

        // Capture the graph
        cudaGraph_t graph;
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        tinyKernel<<<GRID_S, BLOCK, 0, stream>>>(d_s, 1.0001f, N_SMALL);
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));

        // Instantiate
        cudaGraphExec_t instance;
        CUDA_CHECK(cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0));

        // Find the kernel node in the graph (there is only one here)
        cudaGraphNode_t nodes[8];
        size_t numNodes = 8;
        CUDA_CHECK(cudaGraphGetNodes(graph, nodes, &numNodes));
        printf("  Graph has %zu node(s)\n", numNodes);

        // Replay with different scalar each time
        for (int i = 0; i < NUM_SCALARS; i++) {
            // Update the kernel node's parameters (scalar arg in this case
            // requires re-setting the full kernel node params struct)
            cudaKernelNodeParams params = {};
            params.func           = (void*)tinyKernel;
            params.gridDim        = dim3(GRID_S);
            params.blockDim       = dim3(BLOCK);
            params.sharedMemBytes = 0;

            float s = scalar_values[i];
            void* args[] = { &d_s, &s, const_cast<int*>(&N_SMALL) };
            params.kernelParams = args;

            // Find the kernel node (first node that is a kernel node)
            cudaGraphNodeType type;
            for (size_t n = 0; n < numNodes; n++) {
                CUDA_CHECK(cudaGraphNodeGetType(nodes[n], &type));
                if (type == cudaGraphNodeTypeKernel) {
                    CUDA_CHECK(cudaGraphExecKernelNodeSetParams(
                        instance, nodes[n], &params));
                    break;
                }
            }

            CUDA_CHECK(cudaGraphLaunch(instance, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            printf("  Replay %d: scalar=%.4f → launched successfully\n",
                   i, scalar_values[i]);
        }

        CUDA_CHECK(cudaGraphExecDestroy(instance));
        CUDA_CHECK(cudaGraphDestroy(graph));

        printf("\n  Note: you can also update the *contents* of a device buffer\n");
        printf("  between replays by simply writing to the device pointer —\n");
        printf("  the graph records the pointer, not the values behind it.\n");
        printf("  This is the standard pattern for inference: same graph,\n");
        printf("  different input batch written to the same device allocation.\n");
    }

    printf("\n");
    printf("When CUDA Graphs help the most (rule of thumb):\n");
    printf("  kernel_time < 1 ms  → graphs reduce overhead significantly\n");
    printf("  kernel_time > 10 ms → launch overhead is negligible, graphs help little\n\n");
    printf("Profile with Nsight Systems to see the difference:\n");
    printf("  nsys profile --trace=cuda ./build/05_cuda_graphs\n");
    printf("  Compare the 'gaps between kernels' in eager vs graph sections.\n");

    CUDA_CHECK(cudaStreamDestroy(stream));
    cudaFree(d_s); cudaFree(d_l);
    return 0;
}
