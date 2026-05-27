#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// ─────────────────────────────────────────────────────────────────────────────
// CUDA error checking
// ─────────────────────────────────────────────────────────────────────────────

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ─────────────────────────────────────────────────────────────────────────────
// GPU event timer
// ─────────────────────────────────────────────────────────────────────────────

// Usage:
//   GpuTimer t;
//   t.start();
//   kernel<<<...>>>(...);
//   float ms = t.stop_ms();   // synchronises and returns elapsed milliseconds

struct GpuTimer {
    cudaEvent_t _start, _stop;

    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&_start));
        CUDA_CHECK(cudaEventCreate(&_stop));
    }

    ~GpuTimer() {
        cudaEventDestroy(_start);
        cudaEventDestroy(_stop);
    }

    void start() { CUDA_CHECK(cudaEventRecord(_start)); }

    float stop_ms() {
        CUDA_CHECK(cudaEventRecord(_stop));
        CUDA_CHECK(cudaEventSynchronize(_stop));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, _start, _stop));
        return ms;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Print device info
// ─────────────────────────────────────────────────────────────────────────────

inline void print_device_info() {
    int dev;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("Device: %s\n", prop.name);
    printf("  Compute capability : %d.%d\n", prop.major, prop.minor);
    printf("  SMs                : %d\n", prop.multiProcessorCount);
    printf("  Global memory      : %.1f GB\n",
           (double)prop.totalGlobalMem / 1e9);
    printf("  Shared mem / block : %zu KB\n",
           prop.sharedMemPerBlock / 1024);
    printf("  Max threads/block  : %d\n\n", prop.maxThreadsPerBlock);
}
