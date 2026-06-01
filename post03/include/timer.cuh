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

// ─── CUDA event timer ─────────────────────────────────────────────────────────
struct GpuTimer {
    cudaEvent_t _start, _stop;
    GpuTimer()  { CUDA_CHECK(cudaEventCreate(&_start)); CUDA_CHECK(cudaEventCreate(&_stop)); }
    ~GpuTimer() { cudaEventDestroy(_start); cudaEventDestroy(_stop); }
    void  start()      { CUDA_CHECK(cudaEventRecord(_start)); }
    float stop_ms()    {
        CUDA_CHECK(cudaEventRecord(_stop));
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
           p.multiProcessorCount,
           (double)p.totalGlobalMem / 1e9);
}

// ─── Bandwidth helper ─────────────────────────────────────────────────────────
// bytes_moved: total bytes read + written (e.g. 3*n*sizeof(float) for vec add)
// ms         : kernel time in milliseconds
inline double bandwidth_gb_s(size_t bytes_moved, float ms) {
    return (double)bytes_moved / (ms * 1e-3) / 1e9;
}
