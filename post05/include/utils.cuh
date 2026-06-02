#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

// ─── CUDA error checking ──────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d  %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t st = (call);                                             \
        if (st != CUBLAS_STATUS_SUCCESS) {                                      \
            fprintf(stderr, "cuBLAS error at %s:%d  code=%d\n",                \
                    __FILE__, __LINE__, (int)st);                               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ─── GPU event timer ─────────────────────────────────────────────────────────
struct GpuTimer {
    cudaEvent_t _s, _e;
    GpuTimer()  { CUDA_CHECK(cudaEventCreate(&_s)); CUDA_CHECK(cudaEventCreate(&_e)); }
    ~GpuTimer() { cudaEventDestroy(_s); cudaEventDestroy(_e); }
    void  start(cudaStream_t st = 0) { CUDA_CHECK(cudaEventRecord(_s, st)); }
    float stop_ms(cudaStream_t st = 0) {
        CUDA_CHECK(cudaEventRecord(_e, st));
        CUDA_CHECK(cudaEventSynchronize(_e));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, _s, _e));
        return ms;
    }
};

// ─── Device info ─────────────────────────────────────────────────────────────
inline void print_device_info() {
    int dev; CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp p; CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    printf("Device : %s  (sm_%d%d, %d SMs, %.0f GB HBM)\n\n",
           p.name, p.major, p.minor,
           p.multiProcessorCount, (double)p.totalGlobalMem / 1e9);
}

// ─── Section separator ────────────────────────────────────────────────────────
inline void section(const char* t) {
    printf("\n-- %s ", t);
    for (int i = 0; i < 54 - (int)strlen(t); i++) putchar('-');
    printf("\n");
}

// ─── Random float init ────────────────────────────────────────────────────────
inline void rand_fill(float* h, int n, float lo = -0.1f, float hi = 0.1f) {
    for (int i = 0; i < n; i++)
        h[i] = lo + (hi - lo) * (float)rand() / (float)RAND_MAX;
}

// ─── Max absolute difference between two arrays ───────────────────────────────
inline float max_abs_diff(const float* a, const float* b, int n) {
    float d = 0.f;
    for (int i = 0; i < n; i++) d = fmaxf(d, fabsf(a[i] - b[i]));
    return d;
}

// ─── Memory usage report ──────────────────────────────────────────────────────
inline void print_gpu_memory() {
    size_t free_b, total_b;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    printf("  GPU memory: %.1f / %.1f MB used\n",
           (total_b - free_b) / 1e6, total_b / 1e6);
}
