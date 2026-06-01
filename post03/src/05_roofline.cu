// 05_roofline.cu — Post 3: GPU Memory: The Real Bottleneck
//
// Measures where a set of representative kernels sit on the roofline model.
// For each kernel we compute:
//   - Arithmetic intensity (FLOPs / bytes of HBM traffic)
//   - Achieved performance (GFLOP/s or GB/s)
//   - Where it falls relative to the memory bandwidth ceiling
//     and the compute ceiling
//
// Kernels measured:
//   1. Vector scale         — AI ≈ 0.125 FLOPs/byte  (deep memory-bound)
//   2. Vector add           — AI ≈ 0.083 FLOPs/byte  (memory-bound)
//   3. Elementwise GELU     — AI ≈ 3–4 FLOPs/byte    (memory-bound)
//   4. Small matmul (N=64)  — AI ≈ 10 FLOPs/byte     (memory-bound, near ridge)
//   5. Large matmul (N=512) — AI ≈ 85 FLOPs/byte     (compute-bound)
//   6. Large matmul (N=1024)— AI ≈ 170 FLOPs/byte    (solidly compute-bound)
//
// Run this to see the real numbers on your GPU, then compare to the
// roofline diagram in Post 3.
//
// To get the peak bandwidth of your GPU (needed for the roofline ceiling):
//   nvidia-smi --query-gpu=memory.bandwidth --format=csv
// Or look it up from the spec sheet.

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/timer.cuh"

// ─────────────────────────────────────────────────────────────────────────────
// Elementwise kernels (memory-bound)
// ─────────────────────────────────────────────────────────────────────────────

__global__ void vecScale(const float* __restrict__ in, float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * 2.0f;
}

__global__ void vecAdd3(const float* __restrict__ a, const float* __restrict__ b,
                         float* __restrict__ c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
__global__ void geluKernel(const float* __restrict__ in, float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in[i];
        float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
        out[i] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Matrix multiply (compute-bound at large N)
// ─────────────────────────────────────────────────────────────────────────────

#define TILE 16
__global__ void matmulTiled(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float*       __restrict__ C,
                              int N) {
    __shared__ float tA[TILE][TILE], tB[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.f;
    for (int t = 0; t < (N + TILE - 1) / TILE; t++) {
        int ac = t * TILE + threadIdx.x, br = t * TILE + threadIdx.y;
        tA[threadIdx.y][threadIdx.x] = (row < N && ac < N) ? A[row * N + ac] : 0.f;
        tB[threadIdx.y][threadIdx.x] = (br  < N && col < N) ? B[br  * N + col] : 0.f;
        __syncthreads();
        for (int k = 0; k < TILE; k++) sum += tA[threadIdx.y][k] * tB[k][threadIdx.x];
        __syncthreads();
    }
    if (row < N && col < N) C[row * N + col] = sum;
}

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark infrastructure
// ─────────────────────────────────────────────────────────────────────────────

struct RooflinePoint {
    const char* name;
    double ai;            // arithmetic intensity (FLOPs / byte)
    double gflops;        // achieved GFLOP/s
    double bw_gb_s;       // achieved memory bandwidth GB/s (if memory-bound)
    const char* regime;
};

template<typename Fn>
static float bench_kernel(Fn fn, int reps = 30) {
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

    // Query peak hardware limits using the stable cudaDeviceGetAttribute API.
    // cudaDeviceProp.memoryClockRate and .clockRate were removed in newer CUDA.
    int dev; CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    // Memory clock (kHz) and bus width (bits) → peak bandwidth
    int mem_clock_khz = 0, bus_width_bits = 0, core_clock_khz = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&mem_clock_khz,
                                      cudaDevAttrMemoryClockRate, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&bus_width_bits,
                                      cudaDevAttrGlobalMemoryBusWidth, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&core_clock_khz,
                                      cudaDevAttrClockRate, dev));

    // Peak memory bandwidth: 2 (DDR) × clock (Hz) × bus width (bytes)
    double peak_bw_gb_s =
        2.0 * (double)mem_clock_khz * 1e3 * (bus_width_bits / 8) / 1e9;

    // Peak FP32 compute (CUDA cores only — no Tensor Cores):
    // SMs × cores-per-SM × 2 (FMA) × clock.
    // cores-per-SM varies by architecture; we use multiProcessorCount × 128
    // as a conservative lower bound (correct for sm_80/sm_89/sm_90).
    double peak_flops_tflops =
        (double)prop.multiProcessorCount * 128 *
        2.0 * (double)core_clock_khz * 1e3 / 1e12;

    // Ridge point: AI where both ceilings meet
    double ridge = peak_flops_tflops * 1e12 / (peak_bw_gb_s * 1e9);

    printf("Hardware ceilings (from device attributes — CUDA core FP32 only):\n");
    printf("  Memory clock    : %d MHz\n", mem_clock_khz / 1000);
    printf("  Bus width       : %d bits\n", bus_width_bits);
    printf("  Core clock      : %d MHz\n", core_clock_khz / 1000);
    printf("  Peak mem BW     : %.0f GB/s\n", peak_bw_gb_s);
    printf("  Peak FP32       : %.1f TFLOP/s  (no Tensor Cores)\n",
           peak_flops_tflops);
    printf("  Ridge point     : %.1f FLOPs/byte\n\n", ridge);

    int N_elem = 1 << 24;   // 16 M elements for elementwise kernels
    int block  = 256;
    int grid   = (N_elem + block - 1) / block;

    // Allocations for elementwise kernels
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, N_elem * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N_elem * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, N_elem * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_a, 1, N_elem * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_b, 1, N_elem * sizeof(float)));

    printf("%-28s %8s %10s %10s %14s\n",
           "Kernel", "AI", "GFLOP/s", "BW GB/s", "Regime");
    printf("%-28s %8s %10s %10s %14s\n",
           "────────────────────────────","────────","──────────",
           "──────────","──────────────");

    auto report = [&](const char* name, float ms, double flops,
                       size_t hbm_bytes, double ri) {
        double gf  = flops / (ms * 1e-3) / 1e9;
        double bw  = bandwidth_gb_s(hbm_bytes, ms);
        const char* regime = (ri < ridge) ? "memory-bound" : "compute-bound";
        printf("%-28s %7.1f  %10.1f %10.1f %14s\n",
               name, ri, gf, bw, regime);
    };

    // 1. Vector scale: 1 FLOPs/element, 2 × sizeof(float) bytes = 0.125 FLOPs/byte
    float ms = bench_kernel([&](){ vecScale<<<grid,block>>>(d_a, d_c, N_elem); });
    report("vec scale (1 mul)", ms,
           1.0 * N_elem, 2ull * N_elem * sizeof(float), 0.125);

    // 2. Vector add: 1 FLOP/element, 3 × sizeof(float) bytes = 0.083 FLOPs/byte
    ms = bench_kernel([&](){ vecAdd3<<<grid,block>>>(d_a, d_b, d_c, N_elem); });
    report("vec add (a+b)", ms,
           1.0 * N_elem, 3ull * N_elem * sizeof(float), 1.0/12.0);

    // 3. GELU: ~10 FLOPs/element (tanhf ≈ 8-10 FLOPs), 2 × sizeof(float)
    ms = bench_kernel([&](){ geluKernel<<<grid,block>>>(d_a, d_c, N_elem); });
    report("GELU (elementwise)", ms,
           10.0 * N_elem, 2ull * N_elem * sizeof(float), 10.0/(2*4));

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    // 4–6. Matrix multiply at three sizes
    int sizes[] = {64, 512, 1024};
    for (int N : sizes) {
        size_t bytes = (size_t)N * N * sizeof(float);
        float *dA, *dB, *dC;
        CUDA_CHECK(cudaMalloc(&dA, bytes));
        CUDA_CHECK(cudaMalloc(&dB, bytes));
        CUDA_CHECK(cudaMalloc(&dC, bytes));
        CUDA_CHECK(cudaMemset(dA, 1, bytes));
        CUDA_CHECK(cudaMemset(dB, 1, bytes));

        dim3 blk(TILE, TILE);
        dim3 grd((N+TILE-1)/TILE, (N+TILE-1)/TILE);

        ms = bench_kernel([&](){ matmulTiled<<<grd,blk>>>(dA, dB, dC, N); });

        // FLOPs: 2N³ (N² dot products of length N, each 2N FLOPs)
        // HBM bytes: 3 × N² × sizeof(float)  (read A, B; write C)
        double flops = 2.0 * N * N * N;
        double bytes_hbm = 3.0 * N * N * sizeof(float);
        double ai = flops / bytes_hbm;

        char name[32]; snprintf(name, sizeof(name), "matmul N=%d", N);
        report(name, ms, flops, (size_t)bytes_hbm, ai);

        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }

    printf("\n");
    printf("Reading the table\n");
    printf("-----------------\n");
    printf("AI (FLOPs/byte) tells you which ceiling limits your kernel:\n");
    printf("  AI < %.0f  → memory bandwidth ceiling applies\n", ridge);
    printf("              adding more compute units won't help\n");
    printf("  AI > %.0f  → peak FP32 ceiling applies\n", ridge);
    printf("              adding more bandwidth won't help\n\n");
    printf("Most transformer operations (activations, layer norms, attention softmax)\n");
    printf("are deep in the memory-bound regime. The large matmuls are the exception.\n");
    printf("This is why inference decode is so memory-bound (Post 12 of the series).\n\n");
    printf("For a visual roofline plot, run Nsight Compute:\n");
    printf("  ncu --set full ./build/05_roofline\n");
    printf("  (Open the report and navigate to the Speed of Light → Roofline Analysis)\n");

    return 0;
}
