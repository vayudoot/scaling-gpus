# Scaling GPUs — Post 6: Attention on One GPU

Code for **Post 6** of the [Scaling GPUs](https://substack.com/TODO) series.
Implements self-attention from first principles: softmax variants, naive attention
with full memory-traffic analysis, FlashAttention from scratch, and full
multi-head attention with QKV projections.

---

## Structure

```
scaling_gpus_post06/
├── include/
│   └── utils.cuh                    CUDA_CHECK, GpuTimer, rand_fill, section()
├── src/
│   ├── 01_softmax.cu                Naive / stable / online / warp-level softmax
│   ├── 02_attention_naive.cu        Full naive attention + N^2 scaling demo
│   ├── 03_flash_attention.cu        FlashAttention forward from scratch
│   └── 04_multihead_attention.cu    QKV projection + split + per-head attention
├── scripts/
│   └── run_all.sh
└── Makefile
```

**Requires:** CUDA Toolkit with cuBLAS.

```bash
make ARCH=sm_89 && make run
```

---

## Programs

### 1. `01_softmax` — Four softmax implementations

| Version       | Passes | Numerical | Notes                        |
|---------------|--------|-----------|------------------------------|
| Naive         | 2      | Overflows | Never use with real scores   |
| Stable 3-pass | 3      | Correct   | Max subtraction, 3 HBM reads |
| Online 1-pass | 2      | Correct   | Running (max,sum), 2 reads   |
| Warp-level    | 2      | Correct   | shfl_xor, no shared mem      |

**The online softmax algorithm** (the key to FlashAttention):
```
For each new element x_{i+1}:
    m_new = max(m_old, x_{i+1})
    d_new = d_old * exp(m_old - m_new) + exp(x_{i+1} - m_new)
```
The `exp(m_old - m_new)` term rescales the old accumulated sum when the
running maximum increases. Since `m_new >= m_old`, the exponent is always
<= 0 — no overflow possible.

This enables attention to be computed tile by tile: process a chunk of K/V,
update (m, d, O), advance to the next chunk. The full NxN score matrix is
never materialised.

```bash
./build/01_softmax 4096 1024   # 4096 rows, 1024-wide softmax
./build/01_softmax 4096 32     # narrow: warp-level version kicks in
```

---

### 2. `02_attention_naive` — Naive attention with memory analysis

Step-by-step breakdown of `Attention(Q,K,V) = softmax(Q @ K^T / sqrt(d)) @ V`:

```
Step 1+2: S = Q @ K^T / sqrt(d)   [N x N]  compute-bound for large N
Step 3:   P = softmax(S)           [N x N]  deeply memory-bound (AI ~0.375)
Step 4:   O = P @ V                [N x d]  compute-bound for large N
```

**Memory cost of the score matrix S:**

| N    | S matrix  | 8 heads |
|------|-----------|---------|
| 512  |   1 MB    |    8 MB |
| 2048 |  16 MB    |  128 MB |
| 4096 |  64 MB    |  512 MB |
| 8192 | 256 MB    |    2 GB |

The quadratic growth means naive attention OOMs for long sequences well
before you run out of weight memory.

```bash
./build/02_attention_naive
```

**What to verify with Nsight Compute:**
```bash
make ncu-naive-softmax
# Look for dram__bytes.sum — should be ~3 * N * N * 4 bytes for softmax alone
# (3 passes: find max, compute exp, normalise)
```

---

### 3. `03_flash_attention` — FlashAttention forward from scratch

FlashAttention computes `Attention(Q,K,V)` without ever writing the NxN
score matrix to HBM. The algorithm:

```
For each Q-tile (Br rows):
  Load Q_tile into shared memory — stays there the whole inner loop

  For each K/V-tile:
    Load K_tile, V_tile into shared memory
    Compute S_tile = Q_tile @ K_tile^T * scale  [Br x Bc] — stays in registers
    Apply causal mask (S[i,j] = -inf if j > i)
    Online softmax update (per row):
      m_new = max(m_old, rowmax(S_tile))
      O    *= exp(m_old - m_new)        -- rescale old output
      d     = d * exp(m_old - m_new) + rowsum(exp(S_tile - m_new))
      O    += exp(S_tile - m_new) @ V_tile

  Normalise: O /= d
  Write O[Br x d_head] and L[Br] to HBM
```

**HBM traffic comparison (N=2048, d=64, FP32):**
```
Naive:           Q,K reads + write S(16MB) + read S(16MB) + V + O ≈ 33 MB
FlashAttention:  Q,K,V,O only = 4 * 512 KB ≈ 2 MB
Savings:         ~16x at N=2048, growing linearly with N
```

**Tile dimensions in this implementation:**
```
BR = BC = 16   (conservative — fits easily in shared memory)
Shared mem per block: (16 + 16 + 16) * 64 * 4 = 12 KB

Production FlashAttention v2:
  BR = BC = 64 or 128
  Shared mem: ~48 KB — nearly the full SM budget
  Also uses: Tensor Cores (FP16), cp.async for overlap, warp register tiling
```

```bash
./build/03_flash_attention
```

Expected output: correctness PASS and 2–6x speedup vs naive (higher at larger N).

**Profile to see HBM traffic:**
```bash
make ncu-flash
# dram__bytes.sum for flashAttentionFwd should be proportional to N (not N^2)
# Compare to ncu-naive-softmax to see the difference
```

---

### 4. `04_multihead_attention` — Full MHA pipeline

Complete multi-head attention forward pass:

```
Input X [B x S x D]
  |
  | W_QKV [D x 3D] — one fused matmul for all Q, K, V
  v
QKV [B x S x 3D]
  |
  | splitQKVKernel — reshape into heads
  v
Q, K, V [B x H x S x d]    d = D/H
  |
  | per-head attention (naive or flash)
  v
O [B x H x S x d]
  |
  | concat + output projection (W_O [D x D]) — not shown
  v
Output [B x S x D]
```

The fused QKV projection (one SGEMM) vs three separate projections:
- One matmul: `[B*S x D] @ [D x 3D]` — uses cuBLAS with large tiles, high utilisation
- Three matmuls: each `[B*S x D] @ [D x d]` per head — smaller tiles, lower utilisation
- Fused is always faster; it's why `nn.MultiheadAttention` combines them.

**Memory breakdown (B=2, S=512, D=512, H=8):**
```
X input       :  2.1 MB
QKV projected :  6.3 MB
Q, K, V heads :  2.1 MB each
S_buf (attn)  : 16.8 MB  <- eliminated by FlashAttention
O per-head    :  2.1 MB
TOTAL         : 33.5 MB  (18.7 MB without FlashAttention's S_buf)
```

```bash
./build/04_multihead_attention 2 512
./build/04_multihead_attention 4 1024   # larger
```

---

## Key insights from this post

**1. Softmax numerical stability is not optional.**
Attention scores before scaling can be in the hundreds. `exp(100)` is infinity
in float32. Always subtract the row max before exponentiating.

**2. The NxN matrix is the bottleneck, not the matmuls.**
Q@K^T and P@V are compute-bound (high arithmetic intensity) for large N.
Softmax over the NxN matrix is deeply memory-bound (AI ≈ 0.375). FlashAttention's
speedup comes from eliminating the NxN HBM traffic, not from faster matrix math.

**3. Online softmax is the key algorithmic insight.**
Standard softmax needs 2 passes over the row. Online softmax does it in 1.
This enables tiling: process the row in chunks, combining partial statistics.
Without online softmax, you'd need to buffer the entire row before normalising.

**4. FlashAttention is exact, not approximate.**
The output is mathematically identical to naive attention. The online softmax
rescaling ensures the normalisation is correct across all tile boundaries.
There is no approximation — just a different memory access pattern.

---

## Exercises

**Exercise 1 (softmax):** Implement a block-level online softmax where each
block handles multiple rows and uses shared memory for the (max, sum) reduction
across threads. Compare its throughput to the warp-level version on N_COLS=64.

**Exercise 2 (attention):** Profile both naive and flash attention with Nsight
Compute and compare their `dram__bytes.sum` counters. Verify that flash
attention's HBM traffic is O(N) while naive's is O(N^2).

**Exercise 3 (FlashAttention):** Increase the tile sizes to BR=BC=32 in
`03_flash_attention.cu`. You'll need to adjust the shared memory size and
the block dimension. Does the speedup over naive improve?

**Exercise 4 (FlashAttention):** Add causal masking optimisation: for K/V
tiles that are entirely in the future relative to all Q rows in the tile
(i.e., `col_start > row_start + BR`), skip the tile entirely rather than
computing -inf scores. How much does this improve throughput for long sequences?

**Exercise 5 (multi-head):** Replace the naive per-head attention in
`04_multihead_attention.cu` with calls to `flashAttentionFwd` from program 3.
Measure the memory savings (S_buf no longer needed). What is the speedup at
S=2048?

---

## License

MIT.
