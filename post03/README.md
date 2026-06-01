# Scaling GPUs — Post 3: GPU Memory: The Real Bottleneck

Code for **Post 3** of the [Scaling GPUs](https://substack.com/TODO) series.
This post covers the concepts that separate GPU programmers who understand why
their code is fast from those who are merely guessing.

---

## What's in here

```
scaling_gpus_post03/
├── include/
│   └── timer.cuh            CUDA error checking + GpuTimer
├── src/
│   ├── 01_coalescing.cu     Coalesced vs strided vs random memory access
│   ├── 02_transpose.cu      Naive / smem / smem+padding matrix transpose
│   ├── 03_bank_conflicts.cu Shared memory bank conflicts isolated benchmark
│   ├── 04_fusion.cu         Unfused vs fused elementwise kernel pipeline
│   └── 05_roofline.cu       Roofline measurement for multiple kernels
├── scripts/
│   └── run_all.sh
└── Makefile
```

---

## Quick start

```bash
cd scaling_gpus_post03
bash scripts/run_all.sh     # auto-detects GPU, builds, runs everything

# or manually:
make
./build/01_coalescing
./build/02_transpose 4096
./build/03_bank_conflicts
./build/04_fusion
./build/05_roofline
```

Prerequisites and architecture flags are identical to Post 2 — see that README.

---

## Programs

### 1. `01_coalescing` — Memory access pattern benchmark

Five kernels, identical arithmetic, radically different memory access patterns:

| Pattern     | What happens                         | Expected BW vs coalesced |
|-------------|--------------------------------------|--------------------------|
| Stride-1    | 32 consecutive addresses per warp    | 100% (baseline)          |
| Stride-2    | every other address                  | ~50%                     |
| Stride-8    | 8 apart — spans multiple cache lines | ~12%                     |
| Stride-32   | each thread in its own cache line    | ~3%                      |
| Random      | shuffled index, scattered reads      | ~2–5%                    |

The arithmetic intensity of all five kernels is identical (one multiply per
element). The entire throughput difference comes from how many 128-byte
cache-line transactions the hardware must issue per warp.

**To run:**
```bash
./build/01_coalescing           # 4 M elements (default)
./build/01_coalescing 16777216  # 16 M elements
```

**What to verify with Nsight Compute:**
```bash
ncu --metrics dram__bytes.sum \
    --kernel-name strided \
    ./build/01_coalescing_dbg
```
The `dram__bytes.sum` counter for the stride-32 kernel should be ~32× higher
than for the coalesced kernel despite computing the same number of elements.

---

### 2. `02_transpose` — Matrix transpose: three versions

The matrix transpose is the canonical teaching example for both coalescing and
bank conflicts. This program implements and benchmarks all three stages:

**Kernel 1: Naive**
- Reads: coalesced ✓ (consecutive columns per warp)
- Writes: non-coalesced ✗ (writing to rows, N floats apart)
- 32 write transactions per warp → ~32× more bus traffic on writes

**Kernel 2: Shared memory, no padding**
- HBM reads and writes: both coalesced ✓
- Shared memory read: `tile[threadIdx.x][threadIdx.y]` — all 32 threads hit
  bank 0 → 32-way bank conflict ✗
- Faster than naive, but still leaves performance on the table

**Kernel 3: Shared memory, +1 column padding**
- HBM reads and writes: both coalesced ✓
- Shared memory: no conflicts ✓ (each row now starts at a different bank)
- This is the correct implementation. Should approach hardware bandwidth ceiling.

**To run:**
```bash
./build/02_transpose           # default: 4096×4096
./build/02_transpose 1024      # smaller, CPU verification enabled
./build/02_transpose 8192      # larger, more bandwidth pressure
```

**Expected output (A100 80GB, N=4096):**
```
Kernel                           Time (ms)    BW (GB/s)   Speedup
────────────────────────────── ─────────── ──────────── ─────────
Naive (coalesced R, strided W)       5.421       98.4     1.00×
Smem, no padding (bank conflicts)    2.013      265.2     2.69×
Smem, +1 padding (no conflicts)      0.847      630.7     6.40×
```

**The +1 padding insight:**
```cuda
// Slow — 32-way bank conflict on the transposed read
__shared__ float tile[32][32];

// Fast — each row starts at a different bank, zero conflicts
__shared__ float tile[32][32 + 1];  // one extra column per row
```
Cost: 32 extra floats per block (128 bytes). Gain: 6× speedup on this kernel.

**Verify bank conflicts with Nsight Compute:**
```bash
make ncu-transpose-naive    # large conflict count
make ncu-transpose-padded   # should report 0 conflicts
```

---

### 3. `03_bank_conflicts` — Isolated bank conflict benchmark

Strips away all HBM traffic to isolate the shared memory bank conflict effect.
All kernels do the same arithmetic; only the shared memory access pattern changes.

| Kernel              | Bank conflicts | Relative time |
|---------------------|----------------|---------------|
| stride-1            | 0              | 1.0×          |
| stride-16           | 2-way          | ~2.0×         |
| stride-32           | 32-way         | ~5–15×        |
| stride-32 + padding | 0              | ~1.0×         |

**To verify conflict counts with Nsight Compute:**
```bash
make ncu-bank-conflicts
```
Look for `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`:
- `stride-1` and `stride-32 padded` → 0
- `stride-32` → large count proportional to the number of warps

**The bank layout (conceptual):**
```
Bank:     0    1    2   ...  31    0    1   ...
Address: [0]  [4]  [8] ... [124] [128] [132] ...

tile[32][32] — reading column 0:
  tile[0][0]  → address 0   → bank 0
  tile[1][0]  → address 128 → bank 0  ← conflict!
  tile[2][0]  → address 256 → bank 0  ← conflict!
  ... all 32 threads hit bank 0 → 32-way serialisation

tile[32][33] — reading column 0 with +1 padding:
  tile[0][0]  → address 0   → bank 0
  tile[1][0]  → address 132 → bank 1  ← different bank
  tile[2][0]  → address 264 → bank 2  ← different bank
  ... all different → 1 cycle → full bandwidth
```

---

### 4. `04_fusion` — Kernel fusion benchmark

Demonstrates why fusing elementwise operations reduces HBM traffic and
improves throughput.

**The operation chain:** `y = ReLU(LayerNorm(x * weight + bias))`

This is a realistic sequence from the non-matmul part of a transformer MLP block.

| Version         | Kernels | HBM passes | Time   | BW       |
|----------------|---------|------------|--------|----------|
| Unfused        | 3       | 7×         | —      | baseline |
| Fused          | 1       | 2×         | ~3× faster | higher  |

**Why fused is faster:**
```
Unfused:
  x → [scale+bias kernel] → HBM → [layernorm kernel] → HBM → [relu kernel] → y
                            ↑ write                    ↑ read+write
  HBM traffic: 7 × n × sizeof(float)

Fused:
  x → [scale+bias+layernorm+relu kernel] → y
  Intermediates live in shared memory / registers. Never touch HBM.
  HBM traffic: 2 × n × sizeof(float) (input read + output write only)
```

All three individual operations have AI < 1 FLOPs/byte (deeply memory-bound).
Reducing HBM traffic by 3.5× → roughly 3× speedup. This is what
`torch.compile()` does automatically for elementwise op sequences.

**To run:**
```bash
./build/04_fusion              # B=4096 rows, D=1024 features (default)
./build/04_fusion 8192 512     # larger batch, smaller features
```

---

### 5. `05_roofline` — Roofline measurement

Measures real arithmetic intensity and achieved performance for a range of
representative GPU operations and places them on the roofline.

**Sample output (A100 80GB):**
```
Hardware ceilings (approximate):
  Peak memory bandwidth : 1935 GB/s
  Peak FP32 compute     : 19.5 TFLOP/s
  Ridge point           : 10.1 FLOPs/byte

Kernel                       AI     GFLOP/s    BW GB/s  Regime
──────────────────────── ──────── ──────────  ──────── ──────────────
vec scale (1 mul)           0.1       206.4    1651.2  memory-bound
vec add (a+b)               0.1       117.2    1406.5  memory-bound
GELU (elementwise)          1.3       282.2    1129.0  memory-bound
matmul N=64                10.7       685.0      64.1  memory-bound
matmul N=512               85.3     10241.0     120.1  compute-bound
matmul N=1024             170.7     14892.0      87.3  compute-bound
```

**Reading the table:**
- AI < ridge point → memory-bandwidth ceiling applies
- AI > ridge point → peak-FP32 ceiling applies
- vec scale at AI=0.125 is 12× below the ridge → adding Tensor Cores would
  do nothing; only reducing memory traffic helps
- matmul N=1024 at AI=170 is well above the ridge → compute-bound; adding
  bandwidth won't help

**Get the roofline view in Nsight Compute:**
```bash
make ncu-roofline
# Opens as a .ncu-rep file; navigate to Speed of Light → Roofline Analysis
```

---

## Connecting to the post

**The key mental model from this post:**

Every GPU kernel sits somewhere on the roofline. Its position determines which
optimization techniques are worth applying:

```
Memory-bound (AI < ridge):
  ✓ Reduce HBM traffic (fuse kernels, tile into shared memory)
  ✓ Fix coalescing (ensure stride-1 access)
  ✓ Fix bank conflicts (add +1 padding to shared memory tiles)
  ✗ Adding Tensor Cores won't help
  ✗ Increasing occupancy beyond ~50% gives diminishing returns

Compute-bound (AI > ridge):
  ✓ Use Tensor Cores (15× over FP32 CUDA cores)
  ✓ Reduce FLOPs (algorithmic improvements)
  ✗ Fixing memory access patterns won't help much
  ✗ Reducing memory allocations won't help
```

Run `05_roofline` to see where your operations land, then apply the matching
optimization checklist from Post 11 of the series.

---

## Exercises

**Exercise 1 (coalescing):** Modify `01_coalescing.cu` to add a `stride-4`
kernel between stride-2 and stride-8. Does the bandwidth degrade linearly
with stride, or is there a cliff at stride=32 (one cache line per thread)?
This reveals the granularity at which the hardware coalesces.

**Exercise 2 (bank conflicts):** Add a `stride-64` case to `03_bank_conflicts`.
Since 64 % 32 = 0, threads hit the same banks as stride-32. Does the conflict
count match? What about stride-48? (48 % 32 = 16 → 2-way conflict.)

**Exercise 3 (transpose):** Write a fourth transpose kernel that uses a 2D
warp-level cooperative approach instead of shared memory — using `__shfl_sync`
intrinsics to shuffle values between threads in a warp. Compare its performance
to the padded shared memory version. This is how high-performance transpose
kernels actually work in libraries like cuBLAS.

**Exercise 4 (fusion):** Add a second bias vector to `04_fusion` (one per
output feature, added after LayerNorm) and a dropout step (multiply each
element by a Bernoulli sample). Implement the unfused version (4 kernels) and
the fused version. How does the speedup compare to the 3-op version?

**Exercise 5 (roofline):** Pick a kernel from `02_transpose` and compute its
arithmetic intensity analytically:
- How many FLOPs does each thread perform?
- How many bytes of HBM traffic does each thread cause?
Then verify against the Nsight Compute roofline view. Do they match?

---

## Profiling cheat sheet

```bash
# Nsight Systems — full timeline
nsys profile --trace=cuda ./build/02_transpose 4096

# Nsight Compute — bank conflict count
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \
    ./build/03_bank_conflicts

# Nsight Compute — memory throughput vs peak
ncu --metrics \
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    dram__throughput.avg.pct_of_peak_sustained_elapsed \
    ./build/01_coalescing

# Nsight Compute — full roofline
ncu --set full ./build/05_roofline
```

---

## What's next

**Post 4 — CUDA Streams & Async Execution**
Double-buffering pipelines to overlap H2D copies with kernel execution.
The coalescing and bandwidth knowledge from this post explains exactly why
the copy-compute overlap is worth the complexity.

**Post 6 — Attention on One GPU**
FlashAttention is fundamentally a memory-access pattern rewrite of standard
attention — the same tiling + shared-memory reuse principle from Post 3's
transpose, applied to the O(N²) attention matrix. Post 3 is the prerequisite.

---

## License

MIT. Use freely.
