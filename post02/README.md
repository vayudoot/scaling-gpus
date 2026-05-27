# Scaling GPUs — Post 2: Your First CUDA Kernels

Code repository for **Post 2** of the [Scaling GPUs](https://vayudoot.substack.com/p/50be21ae-2c33-4f91-8e06-694009788add)
series. This post covers the CUDA thread model, your first kernels, and the tiled
matrix multiplication that is the foundation of every high-performance GPU operation.

## What's in here

```
post02/
├── include/
│   └── timer.cuh         CUDA error checking + GpuTimer struct
├── src/
│   ├── vec_add.cu        Vector addition (the "Hello World" of CUDA)
│   ├── matmul.cu         Naïve vs tiled matrix multiply with benchmarks
│   └── occupancy.cu      Block-size vs occupancy explorer
├── scripts/
│   └── run_all.sh        One-command build + run for all programs
└── Makefile
```

---

## Prerequisites

| Requirement | Notes |
|---|---|
| NVIDIA GPU | Any Kepler or newer (compute capability 3.5+). Benchmarks shown on A100. |
| CUDA Toolkit 11.8+ | `nvcc --version` to check. Download: [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads) |
| GCC 9+ | Usually installed with the CUDA toolkit |
| GNU Make | Standard on Linux; on Windows use WSL2 |

No Python, no PyTorch, no external libraries. These are pure CUDA C programs.

---

## Quick start

```bash
# Clone or copy this directory, then:
cd post02

# Auto-detect your GPU and run everything
bash scripts/run_all.sh

# Or build manually and run individual programs
make
./build/vec_add
./build/matmul 1024
./build/occupancy
```

### Specifying your GPU architecture

The default `ARCH=native` auto-detects the installed GPU. You can override:

```bash
make ARCH=sm_70   # V100
make ARCH=sm_80   # A100
make ARCH=sm_89   # L4, RTX 4090
make ARCH=sm_90   # H100
```

Using the correct architecture ensures the compiler uses the right instruction
set and Tensor Core generation. `native` is always correct for your local GPU.

---

## Programs

### 1. `vec_add` — Vector addition

The simplest possible CUDA program. Two arrays are added element-wise on the GPU.

**What it teaches:**
- `cudaMalloc` / `cudaMemcpy` / `cudaFree` — the memory management API
- The kernel launch syntax: `kernel<<<gridSize, blockSize>>>(args)`
- Ceiling division for grid size: `(n + blockSize - 1) / blockSize`
- GPU timing with CUDA events (never use CPU timers for GPU kernels)
- Effective memory bandwidth measurement

**Running it:**

```bash
./build/vec_add              # default: 16 M elements
./build/vec_add 67108864     # 64 M elements
```

**Expected output (A100 80GB):**

```
Device: NVIDIA A100-SXM4-80GB
  Compute capability : 8.0
  SMs                : 108
  Global memory      : 80.0 GB
  Shared mem / block : 48 KB
  Max threads/block  : 1024

Vector length : 16777216  (67.1 MB per array)

Launch config : <<<65536 blocks, 256 threads/block>>>

Correctness   : PASS
Kernel time   : 0.384 ms
Bandwidth     : 524.3 GB/s
```

The bandwidth number tells you how close you are to the hardware ceiling
(2,000 GB/s on A100). Vector addition has arithmetic intensity of 0.125 FLOPs/byte
— it's deep in the memory-bound regime. A well-written kernel should approach
~50% of peak bandwidth; higher requires more sophisticated memory access patterns.

---

### 2. `matmul` — Naïve vs tiled matrix multiply

Implements two versions of square matrix multiplication and benchmarks them:

**Naïve:** one thread per output element, all reads from HBM. No data reuse.
Every float loaded from global memory is used once and discarded.

**Tiled:** a block of threads cooperatively loads a 16×16 tile of A and a 16×16
tile of B into shared memory (on-chip, ~32-cycle latency vs ~600-cycle HBM), then
each thread computes its partial dot product against the cached tile. Each float
loaded from HBM is reused 16 times.

**Running it:**

```bash
./build/matmul               # default: 1024 × 1024
./build/matmul 512           # smaller (CPU reference check enabled)
./build/matmul 2048          # larger (shows bigger speedup)
./build/matmul 4096          # large (CPU reference skipped, cross-check only)
```

**Expected output (A100 80GB, N=1024):**

```
──── Naïve matmul ────
  Grid   : (64, 64) blocks of (16, 16) threads
  Time   : 18.43 ms
  Perf   : 116.5 GFLOP/s

──── Tiled matmul (tile=16) ────
  Shared mem / block : 2048 bytes
  Time   : 2.71 ms
  Perf   : 792.1 GFLOP/s

  Speedup (tiled / naïve) : 6.8×

╔══════════════════════════════════════╗
║  Summary (N = 1024)                  ║
╠══════════════════════════════════════╣
║  Naïve  :  116.5 GFLOP/s             ║
║  Tiled  :  792.1 GFLOP/s  (6.8×)     ║
║  Note: cuBLAS reaches ~300+ GFLOP/s  ║
╚══════════════════════════════════════╝
```

The speedup grows with N because larger matrices have higher arithmetic intensity
and the shared memory reuse benefit compounds. At N=2048, you'll typically see
8–12× speedup. At N=4096, 12–20×.

**Why is cuBLAS still much faster?**

Our tiled kernel uses a 16×16 tile, which means each thread handles one output
element with 16 multiplications. Production matmul kernels (cuBLAS, CUTLASS) use:
- Larger register tiles (e.g., each thread computes an 8×8 output block)
- Tensor Core instructions (`wmma` / `mma`) operating on 16×16×16 fragments
- Double-buffering to overlap loading the next tile with computing the current one
- Carefully tuned instruction scheduling to hide latency within a warp

This kernel is pedagogically complete. It explains *why* tiling works. cuBLAS
takes that idea ~20× further with low-level micro-architecture optimisations.

---

### 3. `occupancy` — Block-size vs occupancy explorer

Shows how block size affects theoretical occupancy and measured memory bandwidth.

**What it teaches:**
- `cudaOccupancyMaxPotentialBlockSize` — let CUDA suggest the optimal block size
- `cudaOccupancyMaxActiveBlocksPerMultiprocessor` — compute theoretical occupancy
- Why occupancy matters: more resident warps → better latency hiding
- The practical plateau: beyond ~50% occupancy, bandwidth gains taper off

**Running it:**

```bash
./build/occupancy
```

**Expected output (A100 80GB):**

```
═══ Theoretical occupancy vs block size ═══

Block size | Warps/block | Max blocks/SM | Theor. occupancy
-----------|-------------|---------------|------------------
    32     |      1      |      32       |   50%
    64     |      2      |      32       |  100%
   128     |      4      |      16       |  100%  ← recommended
   256     |      8      |       8       |  100%
   512     |     16      |       4       |  100%
  1024     |     32      |       2       |  100%

═══ Observed bandwidth vs block size ═══

Block size | Kernel time (ms) | Eff. BW (GB/s)
-----------|------------------|----------------
    32     |     1.843 ms     |   145.4
    64     |     0.924 ms     |   290.0
   128     |     0.462 ms     |   580.1
   256     |     0.461 ms     |   581.3
   512     |     0.463 ms     |   578.8
  1024     |     0.462 ms     |   580.0
```

The plateau from block size 128 onward shows that once you have enough occupancy
to hide HBM latency (50–100%), additional warps don't help. Block size 32 is
notably worse because it has only 50% theoretical occupancy — there aren't enough
resident warps to cover memory stalls.

---

## Key concepts from the code

### The indexing formula

```cuda
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

This is the single most-written line in CUDA. It maps each thread's unique 3D
address in the grid to a flat 1D array index. For 2D problems (matrices):

```cuda
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

### The ceiling division pattern

```cuda
int gridSize = (n + blockSize - 1) / blockSize;
```

This ensures you always have enough blocks to cover the entire array, even when
`n` is not a multiple of `blockSize`. The kernel's bounds check (`if (i < n)`)
masks off the threads in the last block that would otherwise go out of bounds.

### `__syncthreads()` in tiled matmul

Two barriers are required, and missing either produces wrong results:

```cuda
// 1. After loading — all threads must finish writing before any reads
__syncthreads();

// ... computation against the loaded tile ...

// 2. After computing — all threads must finish reading before the next
//    iteration overwrites the tile
__syncthreads();
```

`compute-sanitizer --tool racecheck` will catch missing barriers. Try removing
one and running the debug build to see the race condition report.

### Why `__restrict__` matters

```cuda
__global__ void vecAdd(const float* __restrict__ a,
                       const float* __restrict__ b,
                       float*       __restrict__ c, int n)
```

`__restrict__` tells the compiler that the pointers do not alias — `a`, `b`,
and `c` point to non-overlapping memory. This allows the compiler to generate
better load/store instructions and sometimes enables auto-vectorisation. Always
use `__restrict__` on kernel pointer arguments when aliasing is impossible.

---

## Profiling the kernels

### Nsight Systems (timeline)

```bash
# Install Nsight Systems from developer.nvidia.com/nsight-systems
nsys profile --trace=cuda ./build/matmul 2048
# Opens as a .nsys-rep file in the Nsight Systems GUI
```

Look for: kernel execution time, gaps between launches (CPU overhead), and whether
the naïve vs tiled kernels differ in their SM active periods.

### Nsight Compute (per-kernel analysis)

```bash
# Analyse the tiled matmul kernel specifically
ncu --kernel-name matmulTiled --set full ./build/matmul 1024

# Check Tensor Core utilisation (should be 0% for our kernel — we don't use them)
ncu --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active \
    --kernel-name matmulTiled ./build/matmul 1024

# Check memory throughput
ncu --metrics dram__bytes.sum,l1tex__t_bytes.sum \
    --kernel-name matmulTiled ./build/matmul 1024
```

**Roofline interpretation for this kernel (N=1024):**
- Arithmetic intensity ≈ N/6 ≈ 170 FLOPs/byte → compute-bound
- Peak A100 FP32: 19.5 TFLOP/s (CUDA cores only, no Tensor Cores)
- Our tiled kernel achieves ~800 GFLOP/s ≈ 4% of peak

The remaining 96% gap to cuBLAS comes from Tensor Cores (15× speedup on
FP16/BF16) and register-level tiling — topics for Posts 5 and 7.

### compute-sanitizer

```bash
# Build debug versions first
make debug

# Check for out-of-bounds memory access
compute-sanitizer --tool memcheck ./build/matmul_dbg 256

# Check for shared memory race conditions
compute-sanitizer --tool racecheck ./build/matmul_dbg 256
```

---

## Exercises

These are ordered from easiest to most challenging. Aim to complete at least
the first three before reading Post 3.

**Exercise 1:** Change the block size in `vec_add.cu` from 256 to 32, 64, 512,
and 1024. Measure the bandwidth for each. At what block size does performance
plateau? Does this match the occupancy table from `occupancy.cu`?

**Exercise 2:** In `matmul.cu`, change `TILE_SIZE` from 16 to 8 and to 32.
Rebuild and compare performance. Why does a smaller tile produce lower performance?
Why does a larger tile improve performance (up to a point)?

**Exercise 3:** Add a third kernel to `matmul.cu`: a 2D thread mapping where
each thread computes a 2×2 block of output elements (a minimal form of register
tiling). Profile it against the existing tiled kernel. This is the first step
toward cuBLAS-style kernels.

**Exercise 4 (harder):** Implement matrix transpose two ways — naïve (non-coalesced
writes) and shared-memory-buffered (coalesced reads and writes). Measure the
bandwidth for each and use `ncu --metrics dram__bytes.sum` to verify that the
naïve version generates more HBM traffic. This is the canonical example of why
Post 3 (memory coalescing) matters.

**Exercise 5 (harder):** Modify `matmul.cu` to use `__half` (FP16) instead of
`float` and use `nvcuda::wmma` to call Tensor Cores. What is the GFLOP/s now?
How close are you to the hardware ceiling? This is a preview of Post 7.

---

## Connecting to the post

The post explains *why* each pattern works. This code lets you *measure* it.
The most important experiment to run:

```bash
./build/matmul 512   # N=512: CPU reference check enabled
./build/matmul 1024  # N=1024: larger, bigger speedup
./build/matmul 2048  # N=2048: see the speedup grow
```

Watch the GFLOP/s of tiled vs naïve grow with N. At N=512 you'll see ~4–6×.
At N=2048 you'll see ~10–15×. This is the tiling benefit compounding with
arithmetic intensity — exactly what the roofline model in Post 3 predicts.

---

## Troubleshooting

**`nvcc: command not found`**
Add the CUDA toolkit to your PATH: `export PATH=/usr/local/cuda/bin:$PATH`

**`error: cannot find -lcuda`**
Add the CUDA library path: `export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH`

**`no kernel image is available for execution on the device`**
Your `ARCH` flag doesn't match your GPU. Run `nvidia-smi` to see your GPU model,
then find its compute capability and set the correct `sm_XX` flag.

**Results fail correctness check**
This usually means your GPU has a hardware issue (rare) or a driver version
mismatch. Try `compute-sanitizer --tool memcheck ./build/matmul_dbg 256` to check
for memory errors. If the sanitizer also reports errors, file a bug with your
hardware vendor.

**Bandwidth well below expected**
- Check `nvidia-smi` — is the GPU in power-saving mode? Some cloud instances
  throttle idle GPUs. Run any CUDA program first to wake it up.
- Check if ECC is enabled: `nvidia-smi --query-gpu=ecc.mode.current --format=csv`
  ECC reduces effective bandwidth by ~5–10%.
- Check you're testing global memory bandwidth, not shared memory or register
  operations, which are not HBM-limited.

---

## What's next

**Post 3 — GPU Memory: The Real Bottleneck**
Coalescing, bank conflicts, the roofline model. Code will include the matrix
transpose variants mentioned in Exercise 4, benchmarked against each other.

**Post 5 — Building a Neural Net on One GPU**
We take the matmul kernel from this post and build a full MLP: forward pass,
backpropagation, and training loop, using cuBLAS for the matmuls and custom
kernels for activations.

---

## License

GNU General Public License v3.0
