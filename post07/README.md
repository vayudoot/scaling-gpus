# Scaling GPUs -- Post 7: Mixed Precision and Quantization

Code for **Post 7** of the [Scaling GPUs](https://substack.com/TODO) series.
Covers floating-point formats, Tensor Core programming, AMP training, INT8
quantization with dp4a, and W4A16 weight-only quantization for LLM decode.

---

## Structure

```
scaling_gpus_post07/
├── include/
│   └── utils.cuh                 CUDA_CHECK, GpuTimer, rand_fill, fp16 helpers
├── src/
│   ├── 01_precision_formats.cu   FP32/FP16/BF16: overflow, throughput, matmul gap
│   ├── 02_tensor_cores.cu        WMMA API, layout failure demo, benchmarks
│   ├── 03_mixed_precision.cu     AMP: FP32 master + FP16 forward, loss scaling
│   ├── 04_quantization.cu        INT8: per-tensor/channel quant, dp4a matmul
│   └── 05_w4a16.cu               4-bit weights, per-group quant, decode throughput
├── scripts/
│   └── run_all.sh
└── Makefile
```

**Requires:** CUDA Toolkit 11.8+ with cuBLAS.
- BF16 in `01_precision_formats` requires sm_80+ (Ampere+)
- WMMA in `02_tensor_cores` requires sm_70+ (Volta+)
- dp4a in `04_quantization` requires sm_61+ (Pascal+)

```bash
make ARCH=sm_89 && make run    # for RTX 4090 / L4
make ARCH=sm_80 && make run    # for A100
make ARCH=sm_90 && make run    # for H100
```

---

## Programs

### 1. `01_precision_formats` -- Format comparison

Demonstrates why BF16 replaced FP16 for training:

**FP16 overflow:**
```
Value 65505 -> FP16: inf  (max representable: 65504)
Value 1e6   -> FP16: inf
```
Attention scores before scaling can easily exceed 65504, causing NaN cascades.
BF16 keeps FP32's 8-bit exponent: same range (~3.4e38), just less mantissa precision.

**Throughput (memory-bound elementwise):**
```
FP32:  ~1650 GB/s   (4 bytes/elem)
FP16:  ~3300 GB/s   (2 bytes/elem, ~2x)
BF16:  ~3300 GB/s   (same as FP16)
```

**Matmul throughput (Tensor Cores):**
```
FP32 CUDA cores:  ~67 GFLOP/s
FP16 Tensor Cores: ~989 GFLOP/s  (~15x)
```
This is the most important number in GPU ML performance.

**Precision loss in optimizer state:** demonstrates why BF16's 7-bit mantissa
causes gradient updates to round to zero when added to large weights --
the core reason FP32 master weights are needed for training.

### 2. `02_tensor_cores` -- WMMA API

Implements matrix multiplication using raw WMMA intrinsics.
Each `mma_sync()` call processes a 16x16x16 tile = 8192 FLOPs in one instruction.

**The layout requirement:**
```cuda
// Fragment A must be row_major
wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fragA;

// Fragment B must be col_major when B is stored [K x N]
wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fragB;
```

Using `row_major` for fragB when B is stored `[K x N]` row-major produces
**silently wrong results** -- no error, no NaN, just incorrect output. This
is the most dangerous silent failure in Tensor Core programming. The demo
kernel shows this happening live.

**Performance:**
```
FP32 CUDA cores:       ~67 GFLOP/s    1.0x
Our WMMA kernel:       ~350 GFLOP/s   ~5x
cuBLAS FP16 (optimised): ~989 GFLOP/s  ~15x
```
The gap between WMMA and cuBLAS: cuBLAS uses register-level tiling, pipelined
memory loads (cp.async), and hand-tuned inner loops. WMMA shows the programming
model; use cuBLAS or CUTLASS for production code.

**Verify Tensor Core utilization:**
```bash
make ncu-tensor-core-util
# cuBLAS FP16 should show >50% tensor pipe utilization
# cuBLAS FP32 should show ~0%
```

### 3. `03_mixed_precision` -- AMP training

The AMP contract:
```
FP32 master weights  -->  cast to FP16  -->  forward (Tensor Cores)
                                              backward (Tensor Cores)
                                         <--  accumulate gradients in FP32
FP32 master weights  <--  SGD/Adam update
```

Memory per parameter in AMP training:
```
FP32 master weight:    4 bytes
FP16 working copy:     2 bytes
FP32 gradient:         4 bytes
FP32 Adam m:           4 bytes
FP32 Adam v:           4 bytes
Total:                18 bytes  (vs 4 bytes inference)
```

**Loss scaling:** FP16 minimum positive normal ~6e-8. Gradients smaller than
this underflow to zero, stalling training. Loss scaling multiplies the loss
by a constant (2^15) before backward, shifts gradients into FP16 range, then
divides them back after backward. BF16 doesn't need this.

### 4. `04_quantization` -- INT8 with dp4a

**Per-tensor vs per-channel accuracy:**
```
Per-tensor INT8:  max_err ~0.08  (outliers in one channel corrupt all scales)
Per-channel INT8: max_err ~0.004  (each channel gets its own scale)
```
Per-channel quantization scales each output row (channel) independently.
The overhead is minimal: one extra float per output channel.

**dp4a instruction:**
```cuda
// Computes c += b0*a0 + b1*a1 + b2*a2 + b3*a3 in one instruction
// a, b are four INT8 values packed into INT32
int acc = __dp4a(a_pack, b_pack, acc);
```
Four INT8 multiply-adds per instruction vs one FP32 FMA = 4x more arithmetic.
Combined with 4x less memory traffic: INT8 throughput is ~4-8x FP32.

### 5. `05_w4a16` -- 4-bit weight quantization

**Why W4A16 dominates LLM inference:**

During decode (batch=1), the GPU loads the entire weight matrix for one
vector-matrix multiply. Arithmetic intensity = 2 / bytes_per_weight:

```
FP16:  AI = 1.0 FLOPs/byte  -> bounded by 3350 GB/s = 3350 GFLOP/s
INT4:  AI = 4.0 FLOPs/byte  -> bounded by 3350 GB/s = 13400 GOPS (4x)
```

**Per-group quantization (G=128):**
One scale per 128 consecutive weights. Vastly more accurate than per-tensor
INT4 because local distributions can be captured per-group:
```
Per-tensor INT4:   ~5-10% relative error
Per-channel INT4:  ~2-5% relative error
Per-group G=128:   ~0.5-1% relative error  (near-FP16 quality)
```
This is the scheme used by GPTQ, AWQ, and bitsandbytes 4-bit.

**Memory savings:**
```
70B model at FP16:  140 GB  (needs 2 x H100)
70B model at INT4:   35 GB  (fits on 1 x H100)
```

**Note on kernel performance:** The `w4a16MatmulKernel` in program 5 is a
pedagogical implementation showing the dequant-then-multiply pattern. Production
W4A16 kernels (in llama.cpp, GPTQ-CUDA, AWQ) use:
- Tiled shared memory loading
- Vectorized 8-value INT4 unpacking via `ldg.128`
- Tensor Core MMA per tile (in GPTQ-MARLIN, vLLM)
- Achieves 3-4x FP16 throughput at batch=1 on H100

---

## The precision landscape

```
Format  Bytes  Exponent  Mantissa  Max value     Best for
------  -----  --------  --------  ----------    --------
FP64     8      11        52        ~1.8e308     Scientific (rarely ML)
FP32     4       8        23        ~3.4e38      Optimizer state, reference
BF16     2       8         7        ~3.4e38      Training default (Ampere+)
FP16     2       5        10        ~65504       Inference, older hardware
INT8     1       --        --        127         W8A8 inference
INT4    0.5      --        --          7         W4A16 LLM decode
```

The key insight: **BF16 has the same exponent as FP32** (8 bits). Same range,
less mantissa precision. This is why BF16 doesn't need loss scaling: gradients
can't overflow no matter how large.

---

## Exercises

**Exercise 1 (precision):** Time a large softmax (N=4096x4096) in FP32 vs FP16.
Since softmax is memory-bound, the FP16 version should be ~2x faster. Verify
the outputs match to within 1e-3 relative error. Does FP16 produce any NaN
for large input values?

**Exercise 2 (WMMA):** Extend the WMMA kernel to use BF16 inputs instead of FP16.
Replace `__half` with `__nv_bfloat16` in the fragment declarations. Run on the
same inputs and verify the correctness check still passes. Compare throughput.

**Exercise 3 (quantization):** Implement per-group INT8 quantization (G=128,
scale per group of 128 weights per row). Compare accuracy to per-channel INT8
on a weight matrix with outliers. When does per-group INT8 significantly improve
over per-channel?

**Exercise 4 (W4A16):** Modify `w4a16MatmulKernel` to use shared memory tiling:
load a tile of packed INT4 weights into shared memory, unpack and dequantize
collectively, then compute the output tile. How does this improve throughput?

**Exercise 5 (loss scaling):** Implement a dynamic loss scaler in CUDA that
runs the overflow check kernel after each backward pass and adjusts the scale
accordingly. Simulate 100 training steps and plot the scale value over time.

---

## License

MIT.
