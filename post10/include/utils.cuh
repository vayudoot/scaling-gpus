#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>

// ─── Error checking ───────────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(err_));             \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t st_ = (call);                                           \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                    \
            fprintf(stderr, "cuBLAS error %s:%d  code=%d\n",                  \
                    __FILE__, __LINE__, (int)st_);                            \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// ─── GPU event timer ─────────────────────────────────────────────────────────
struct GpuTimer {
    cudaEvent_t _start, _stop;
    GpuTimer()  { CUDA_CHECK(cudaEventCreate(&_start)); CUDA_CHECK(cudaEventCreate(&_stop)); }
    ~GpuTimer() { cudaEventDestroy(_start); cudaEventDestroy(_stop); }
    void  start(cudaStream_t s = 0) { CUDA_CHECK(cudaEventRecord(_start, s)); }
    float stop_ms(cudaStream_t s = 0) {
        CUDA_CHECK(cudaEventRecord(_stop, s));
        CUDA_CHECK(cudaEventSynchronize(_stop));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, _start, _stop));
        return ms;
    }
};

// ─── Device info ─────────────────────────────────────────────────────────────
inline void print_device_info() {
    int n = 0; CUDA_CHECK(cudaGetDeviceCount(&n));
    printf("Detected %d CUDA device%s\n", n, n == 1 ? "" : "s");
    int dev; CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp p; CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    printf("Using  : %s (sm_%d%d, %.0f GB HBM)\n\n",
           p.name, p.major, p.minor, (double)p.totalGlobalMem / 1e9);
}

// ─── Section separator ────────────────────────────────────────────────────────
inline void section(const char* t) {
    printf("\n-- %s ", t);
    for (int i = 0; i < 54 - (int)strlen(t); i++) putchar('-');
    printf("\n");
}

// ─── Helpers ─────────────────────────────────────────────────────────────────
inline void rand_fill(float* p, int n, float lo = -0.1f, float hi = 0.1f) {
    for (int i = 0; i < n; i++)
        p[i] = lo + (hi - lo) * (float)rand() / (float)RAND_MAX;
}

inline float max_abs_diff(const float* a, const float* b, int n) {
    float d = 0.f;
    for (int i = 0; i < n; i++) d = fmaxf(d, fabsf(a[i] - b[i]));
    return d;
}

inline double bw_gb_s(size_t bytes, float ms) {
    return (double)bytes / (ms * 1e-3) / 1e9;
}
