// 01_activations.cu — Post 5: Building a Neural Network on One GPU
//
// Activation functions are the simplest neural network kernels.
// Each one is a pure elementwise operation: one input float, one output float.
//
// This file implements and benchmarks:
//   - ReLU, GELU, sigmoid, tanh — the four most common activations
//   - Unfused vs fused bias+activation (showing the HBM round-trip cost)
//   - Backward passes (gradients) for ReLU and GELU
//
// Key lesson: all elementwise kernels are MEMORY-BOUND.
// Arithmetic intensity = (FLOPs per element) / (bytes read + written).
// For ReLU: 1 FLOP / 8 bytes = 0.125 FLOPs/byte — far below the ridge point.
// Fusing bias addition with the activation saves one full HBM read+write.

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include "../include/utils.cuh"

// ─────────────────────────────────────────────────────────────────────────────
// Forward activation kernels
// ─────────────────────────────────────────────────────────────────────────────

__global__ void reluKernel(const float* __restrict__ in,
                            float*       __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = fmaxf(0.f, in[i]);
}

// GELU approximation (Hendrycks & Gimpel 2016), same as used in GPT-2/BERT.
// 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
__global__ void geluKernel(const float* __restrict__ in,
                            float*       __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in[i];
        float c = 0.7978845608f * (x + 0.044715f * x * x * x);
        out[i]  = 0.5f * x * (1.f + tanhf(c));
    }
}

__global__ void sigmoidKernel(const float* __restrict__ in,
                               float*       __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 1.f / (1.f + expf(-in[i]));
}

__global__ void tanhKernel(const float* __restrict__ in,
                            float*       __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = tanhf(in[i]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Fused bias + activation kernels
// ─────────────────────────────────────────────────────────────────────────────
// Adding bias is also an elementwise op. Running it as a separate kernel
// means: read activations from HBM, write biased activations to HBM, then
// read them again for ReLU, write again. Two full round-trips for trivial work.
//
// Fused: read once, apply bias+activation, write once. Half the HBM traffic.
// For a 256 MB activation tensor this saves 512 MB of HBM transfers.

__global__ void fusedBiasReluKernel(const float* __restrict__ in,
                                     const float* __restrict__ bias,
                                     float*       __restrict__ out,
                                     int n_rows, int n_cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_rows * n_cols;
    if (idx >= total) return;
    int col = idx % n_cols;
    out[idx] = fmaxf(0.f, in[idx] + bias[col]);
}

__global__ void fusedBiasGeluKernel(const float* __restrict__ in,
                                     const float* __restrict__ bias,
                                     float*       __restrict__ out,
                                     int n_rows, int n_cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_rows * n_cols) return;
    int col = idx % n_cols;
    float x = in[idx] + bias[col];
    float c = 0.7978845608f * (x + 0.044715f * x * x * x);
    out[idx] = 0.5f * x * (1.f + tanhf(c));
}

// ─────────────────────────────────────────────────────────────────────────────
// Backward passes
// ─────────────────────────────────────────────────────────────────────────────
// The backward pass gates the upstream gradient through the activation's
// derivative.  For ReLU: derivative is 1 where x > 0, else 0.
// We need the FORWARD INPUT (pre-activation) to compute this.
// This is why forward activations must be SAVED during the forward pass.

__global__ void reluBackward(const float* __restrict__ fwd_input,
                              const float* __restrict__ grad_out,
                              float*       __restrict__ grad_in,
                              int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Pass gradient through only where the forward input was positive.
    // gate = (fwd_input[i] > 0) ? 1 : 0
    if (i < n) grad_in[i] = (fwd_input[i] > 0.f) ? grad_out[i] : 0.f;
}

// GELU backward: d/dx GELU(x) = 0.5*tanh(c) + (0.5*x*(1-tanh²(c)))*c'
// where c = sqrt(2/pi)*(x + 0.044715*x^3), c' = sqrt(2/pi)*(1 + 3*0.044715*x^2)
__global__ void geluBackward(const float* __restrict__ fwd_input,
                              const float* __restrict__ grad_out,
                              float*       __restrict__ grad_in,
                              int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x  = fwd_input[i];
    float c  = 0.7978845608f * (x + 0.044715f * x * x * x);
    float tc = tanhf(c);
    float c_ = 0.7978845608f * (1.f + 3.f * 0.044715f * x * x);
    // d/dx GELU(x) = 0.5*(1+tanh(c)) + 0.5*x*(1-tanh²(c))*c'
    grad_in[i] = grad_out[i] * (0.5f + 0.5f * tc + 0.5f * x * (1.f - tc * tc) * c_);
}

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark helper
// ─────────────────────────────────────────────────────────────────────────────
template<typename Fn>
static float bench(Fn fn, int reps = 50) {
    fn(); CUDA_CHECK(cudaDeviceSynchronize());
    GpuTimer t; t.start();
    for (int r = 0; r < reps; r++) fn();
    return t.stop_ms() / reps;
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main() {
    print_device_info();
    srand(42);

    const int ROWS  = 4096;
    const int COLS  = 4096;
    const int N     = ROWS * COLS;   // 16 M elements = 64 MB
    const int BLOCK = 256;
    const int GRID  = (N + BLOCK - 1) / BLOCK;

    // Host setup
    float* h_in  = (float*)malloc(N * sizeof(float));
    float* h_bias = (float*)malloc(COLS * sizeof(float));
    rand_fill(h_in,  N,    -2.f, 2.f);
    rand_fill(h_bias, COLS, -0.1f, 0.1f);

    // Device buffers
    float *d_in, *d_out, *d_tmp, *d_bias, *d_grad_out, *d_grad_in;
    CUDA_CHECK(cudaMalloc(&d_in,       N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out,      N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tmp,      N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias,     COLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_out, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_in,  N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_in,   h_in,   N * sizeof(float),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, h_bias, COLS * sizeof(float),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_grad_out, 1, N * sizeof(float)));

    size_t bytes2 = 2ull * N * sizeof(float);   // read + write

    // ── Forward activations ──────────────────────────────────────────────────
    section("Forward activation benchmarks");
    printf("  (N = %d x %d = %d elements = %.0f MB per tensor)\n\n",
           ROWS, COLS, N, (double)N * 4 / 1e6);

    printf("  %-25s %10s %10s %12s\n", "Kernel", "Time(ms)", "BW(GB/s)", "AI(FLOP/B)");
    printf("  %-25s %10s %10s %12s\n",
           "-------------------------", "--------", "--------", "----------");

    auto report = [&](const char* name, float ms, double flops_per_elem) {
        double bw  = (double)bytes2 / (ms * 1e-3) / 1e9;
        double ai  = flops_per_elem / 8.0;  // 8 bytes per elem (read+write, float32)
        printf("  %-25s %10.3f %10.1f %12.3f\n", name, ms, bw, ai);
    };

    float ms;
    ms = bench([&](){ reluKernel   <<<GRID,BLOCK>>>(d_in, d_out, N); });
    report("ReLU",    ms, 1.0);

    ms = bench([&](){ sigmoidKernel<<<GRID,BLOCK>>>(d_in, d_out, N); });
    report("Sigmoid", ms, 4.0);  // exp + add + div + reciprocal

    ms = bench([&](){ tanhKernel   <<<GRID,BLOCK>>>(d_in, d_out, N); });
    report("Tanh",    ms, 4.0);

    ms = bench([&](){ geluKernel   <<<GRID,BLOCK>>>(d_in, d_out, N); });
    report("GELU",    ms, 10.0);

    // ── Unfused vs fused bias + ReLU ─────────────────────────────────────────
    section("Unfused vs fused: bias + ReLU");
    printf("  Unfused: 2 kernels, 2 HBM round-trips\n");
    printf("  Fused:   1 kernel,  1 HBM round-trip\n\n");

    // Unfused: separate bias add + ReLU kernels
    // (Using a simple elementwise bias broadcast — one kernel for illustration)
    auto addBias = [&]() {
        // Each thread adds bias[col] to d_tmp[row*COLS+col]
        // (reusing fusedBiasReluKernel idea but split into two steps via tmp)
        fusedBiasReluKernel<<<GRID,BLOCK>>>(d_in, d_bias, d_tmp, ROWS, COLS);
    };
    auto applyRelu = [&]() { reluKernel<<<GRID,BLOCK>>>(d_tmp, d_out, N); };

    ms = bench([&](){ addBias(); applyRelu(); });
    printf("  %-25s %.3f ms  (2x HBM: %.0f MB read + %.0f MB write)\n",
           "Unfused bias+ReLU", ms,
           (double)N * 4 / 1e6, (double)N * 4 / 1e6);

    ms = bench([&](){ fusedBiasReluKernel<<<GRID,BLOCK>>>(d_in,d_bias,d_out,ROWS,COLS); });
    printf("  %-25s %.3f ms  (1x HBM: %.0f MB read + %.0f MB write)\n",
           "Fused  bias+ReLU", ms,
           (double)N * 4 / 1e6, (double)N * 4 / 1e6);

    ms = bench([&](){ fusedBiasGeluKernel<<<GRID,BLOCK>>>(d_in,d_bias,d_out,ROWS,COLS); });
    printf("  %-25s %.3f ms\n", "Fused  bias+GELU", ms);

    // ── Backward passes ──────────────────────────────────────────────────────
    section("Backward passes (gradient kernels)");
    printf("  These need the SAVED forward input — this is activation memory.\n\n");

    ms = bench([&](){ reluBackward<<<GRID,BLOCK>>>(d_in, d_grad_out, d_grad_in, N); });
    printf("  %-25s %.3f ms\n", "ReLU backward", ms);

    ms = bench([&](){ geluBackward<<<GRID,BLOCK>>>(d_in, d_grad_out, d_grad_in, N); });
    printf("  %-25s %.3f ms\n", "GELU backward", ms);

    printf("\n  Note: backward kernels read fwd_input + grad_out, write grad_in\n");
    printf("  = 3 tensor passes = 3x the HBM traffic of a simple elementwise op.\n");
    printf("  This is why saving activations costs memory AND why recomputing\n");
    printf("  them (gradient checkpointing) trades extra FLOPs for less memory.\n");

    // Cleanup
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_tmp);
    cudaFree(d_bias); cudaFree(d_grad_out); cudaFree(d_grad_in);
    free(h_in); free(h_bias);
    return 0;
}
