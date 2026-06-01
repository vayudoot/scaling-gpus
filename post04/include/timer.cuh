#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// ─── CUDA error checking ──────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ─── Wall-clock timer (CPU side) ─────────────────────────────────────────────
#include <time.h>
static inline double wall_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

// ─── GPU event timer ─────────────────────────────────────────────────────────
struct GpuTimer {
    cudaEvent_t _start, _stop;
    GpuTimer()  { CUDA_CHECK(cudaEventCreate(&_start));
                  CUDA_CHECK(cudaEventCreate(&_stop)); }
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
    int dev; CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp p; CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    printf("Device : %s  (sm_%d%d, %d SMs, %.0f GB HBM, %d copy engine%s)\n\n",
           p.name, p.major, p.minor, p.multiProcessorCount,
           (double)p.totalGlobalMem / 1e9,
           p.asyncEngineCount,
           p.asyncEngineCount == 1 ? "" : "s");
}

// ─── Bandwidth helper ─────────────────────────────────────────────────────────
inline double bandwidth_gb_s(size_t bytes, float ms) {
    return (double)bytes / (ms * 1e-3) / 1e9;
}

// ─── Simple section separator ────────────────────────────────────────────────
inline void section(const char* title) {
    printf("\n-- %s ", title);
    int len = 55 - (int)strlen(title);
    for (int i = 0; i < len; i++) putchar('-');
    printf("\n");
}
