#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>

// ─── Error checking ───────────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t e = (call);                                                 \
        if (e != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(e));                 \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t s = (call);                                              \
        if (s != CUBLAS_STATUS_SUCCESS) {                                       \
            fprintf(stderr, "cuBLAS error %s:%d  code=%d\n",                   \
                    __FILE__, __LINE__, (int)s);                                \
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

// ─── Helpers ─────────────────────────────────────────────────────────────────
inline void rand_fill_fp32(float* p, int n, float lo = -1.f, float hi = 1.f) {
    for (int i = 0; i < n; i++)
        p[i] = lo + (hi - lo) * (float)rand() / (float)RAND_MAX;
}

inline float max_abs_diff_fp32(const float* a, const float* b, int n) {
    float d = 0.f;
    for (int i = 0; i < n; i++) d = fmaxf(d, fabsf(a[i] - b[i]));
    return d;
}

inline double bw_gb_s(size_t bytes, float ms) {
    return (double)bytes / (ms * 1e-3) / 1e9;
}

inline double gflops(double flops, float ms) {
    return flops / (ms * 1e-3) / 1e9;
}

// ─── FP16 conversion helpers ─────────────────────────────────────────────────
inline void fp32_to_fp16(const float* src, __half* dst, int n) {
    for (int i = 0; i < n; i++) dst[i] = __float2half(src[i]);
}

inline void fp16_to_fp32(const __half* src, float* dst, int n) {
    for (int i = 0; i < n; i++) dst[i] = __half2float(src[i]);
}
