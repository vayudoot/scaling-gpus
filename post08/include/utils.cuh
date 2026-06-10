#pragma once
#include <cuda_runtime.h>
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

// ─── GPU event timer ─────────────────────────────────────────────────────────
struct GpuTimer {
    cudaEvent_t _start, _stop;
    GpuTimer()  { CUDA_CHECK(cudaEventCreate(&_start)); CUDA_CHECK(cudaEventCreate(&_stop)); }
    ~GpuTimer() { cudaEventDestroy(_start); cudaEventDestroy(_stop); }
    void  start(cudaStream_t st = 0) { CUDA_CHECK(cudaEventRecord(_start, st)); }
    float stop_ms(cudaStream_t st = 0) {
        CUDA_CHECK(cudaEventRecord(_stop, st));
        CUDA_CHECK(cudaEventSynchronize(_stop));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, _start, _stop));
        return ms;
    }
};

// ─── Device info ─────────────────────────────────────────────────────────────
inline void print_device_info() {
    int n = 0; CUDA_CHECK(cudaGetDeviceCount(&n));
    printf("Detected %d CUDA device%s:\n", n, n == 1 ? "" : "s");
    for (int d = 0; d < n; d++) {
        cudaDeviceProp p; CUDA_CHECK(cudaGetDeviceProperties(&p, d));
        printf("  [%d] %s  (sm_%d%d, %d SMs, %.0f GB HBM)\n",
               d, p.name, p.major, p.minor,
               p.multiProcessorCount, (double)p.totalGlobalMem / 1e9);
    }
    printf("\n");
}

// ─── Section separator ────────────────────────────────────────────────────────
inline void section(const char* t) {
    printf("\n-- %s ", t);
    for (int i = 0; i < 54 - (int)strlen(t); i++) putchar('-');
    printf("\n");
}

// ─── Helpers ─────────────────────────────────────────────────────────────────
inline void rand_fill(float* p, int n, float lo = -1.f, float hi = 1.f) {
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
