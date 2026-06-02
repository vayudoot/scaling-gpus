# Scaling GPUs — Post 5: Building a Neural Network on One GPU

Code for **Post 5** of the [Scaling GPUs](https://substack.com/TODO) series.
Implements a complete two-layer MLP from scratch: forward pass, backpropagation,
and a training loop with SGD and Adam — using cuBLAS for matmuls and custom
CUDA kernels for everything else.

---

## Structure

```
scaling_gpus_post05/
├── include/
│   └── utils.cuh           CUDA_CHECK, CUBLAS_CHECK, GpuTimer, rand_fill
├── src/
│   ├── 01_activations.cu   ReLU / GELU / sigmoid / tanh + fused bias+act
│   ├── 02_mlp_forward.cu   Two-layer MLP forward pass + tiled vs cuBLAS comparison
│   ├── 03_backprop.cu      Full backward pass — every gradient kernel explicit
│   └── 04_training_loop.cu Complete training loop with Adam on synthetic data
├── scripts/
│   └── run_all.sh
└── Makefile
```

**Requires:** CUDA Toolkit with cuBLAS (standard in any CUDA install).

```bash
make && make run       # build and run everything
make ARCH=sm_89 run    # for RTX 4090 / L4
```

---

## Programs

### 1. `01_activations` — Activation functions as CUDA kernels

Benchmarks four activation functions and shows why fusing bias addition with
the activation saves a full HBM round-trip.

| Kernel         | AI (FLOPs/byte) | Regime        |
|----------------|-----------------|---------------|
| ReLU           | 0.125           | memory-bound  |
| Sigmoid        | 0.5             | memory-bound  |
| GELU           | 1.25            | memory-bound  |
| Fused bias+ReLU| 0.125           | memory-bound (1 pass instead of 2) |

All elementwise kernels are memory-bound. Fusion doesn't change the regime —
it reduces the number of HBM passes from 2 to 1, directly halving runtime.

The backward kernels (ReLU and GELU) read three tensors (forward input, upstream
gradient, output gradient) instead of two — 50% more HBM traffic than forward.
This is unavoidable because the gradient depends on the forward input.

```bash
./build/01_activations
ncu --set full --kernel-name geluKernel ./build/01_activations
```

---

### 2. `02_mlp_forward` — Two-layer MLP forward pass

Architecture: `out = ReLU(x @ W1^T + b1) @ W2^T + b2`

**cuBLAS layout convention:** cuBLAS uses column-major storage, but our tensors
are row-major. The standard trick for row-major A[M,K] @ B[K,N] = C[M,N] is:
```cuda
// Swap A and B, use CUBLAS_OP_T on the first argument:
cublasSgemm(handle,
    CUBLAS_OP_T, CUBLAS_OP_N,   // transpose W, not x
    N, M, K,                     // note: N comes first (cuBLAS is col-major)
    &alpha,
    W, K,                        // W[N x K] passed as col-major [K x N]
    x, K,                        // x[M x K]
    &beta,
    out, N);                     // output [M x N]
```
This produces the correct result without physically transposing either matrix.

**tiled kernel vs cuBLAS comparison** (A100, B=512, Din=1024, Dhid=2048):
```
Tiled kernel  :  ~120 GFLOP/s   (Post 2 implementation)
cuBLAS SGEMM  : ~6000 GFLOP/s   (Tensor Cores + aggressive register tiling)
cuBLAS speedup:    ~50x
```

The gap exists because cuBLAS:
1. Uses Tensor Core warp-level MMA instructions (wmma / ptx mma)
2. Tiles registers instead of shared memory (each thread accumulates an 8×8 block)
3. Uses asynchronous global memory loads (cp.async) to hide HBM latency
4. Has hand-tuned inner loops for each SM generation

Our tiled kernel is pedagogically correct and shows the principle. cuBLAS takes
the same principle ~20× further with low-level micro-architecture optimisation.

```bash
./build/02_mlp_forward 512    # batch=512
./build/02_mlp_forward 64     # small batch — CPU reference check enabled
```

---

### 3. `03_backprop` — Backpropagation from scratch

Every backward operation written out explicitly:

```
Forward:   h_pre = x @ W1^T + b1     [B x Dhid]  <- SAVED
           h     = ReLU(h_pre)        [B x Dhid]  <- SAVED
           out   = h @ W2^T + b2      [B x Dout]

Backward:
  Step 1:  dL/dout  = 2*(out-target)/N          [B x Dout]    mseLossGrad kernel
  Step 2:  dL/dW2   = h^T @ dL/dout             [Dhid x Dout] cublasSgemm
  Step 3:  dL/db2   = sum_rows(dL/dout)          [Dout]        biasGrad kernel
  Step 4:  dL/dh    = dL/dout @ W2              [B x Dhid]    cublasSgemm
  Step 5:  dL/dh_pre = dL/dh * (h_pre > 0)      [B x Dhid]    reluBwd kernel
  Step 6:  dL/dW1   = x^T @ dL/dh_pre           [Din x Dhid]  cublasSgemm
  Step 7:  dL/db1   = sum_rows(dL/dh_pre)        [Dhid]        biasGrad kernel
```

**The 2× backward rule:** each forward matmul generates two backward matmuls
(one for dW, one for dX). Backward pass is ~2× the compute of forward.
Measured ratio on A100 is typically 1.8–2.2×.

**Memory breakdown** (B=256, Din=512, Dhid=1024, Dout=128):
```
Parameters        :   2.2 MB
Saved activations :   3.1 MB  (x, h_pre, h — needed for backward)
Weight gradients  :   2.2 MB
Activation grads  :   1.2 MB
TOTAL training    :   8.7 MB
Inference only    :   2.2 MB  (3.9x less)
```

```bash
./build/03_backprop 256
./build/03_backprop 32   # small batch — runs CPU correctness check
```

---

### 4. `04_training_loop` — Full training with Adam

Trains a toy MLP on a synthetic binary classification task. The loss should
decrease visibly over 500 steps.

**What this demonstrates:**

**Gradient zeroing** — the most common training bug for beginners:
```cuda
// WRONG: gradients accumulate from previous steps
forward(); backward(); optimizer_step();

// CORRECT: zero gradients before each backward pass
forward(); backward(); optimizer_step();
cudaMemset(d_dW1, 0, ...);  // zero ALL gradient buffers
cudaMemset(d_dW2, 0, ...);
// etc.
```
PyTorch calls this `optimizer.zero_grad()`. Forgetting it causes gradients to
accumulate over steps, producing incorrect (and often diverging) updates.

**Adam optimizer kernel:**
```cuda
float mi = beta1 * m[i] + (1.f - beta1) * g[i];   // momentum
float vi = beta2 * v[i] + (1.f - beta2) * g[i]*g[i]; // variance
m[i] = mi; v[i] = vi;
float m_hat = mi / (1 - beta1^t);   // bias correction
float v_hat = vi / (1 - beta2^t);
p[i] -= lr * m_hat / (sqrt(v_hat) + eps);
```
Adam reads 4 arrays (param, grad, m, v) and writes 3 (param, m, v).
7 tensor passes → deeply memory-bound, like all optimizer steps.
Adam adds 2× parameter memory for the optimizer state (m and v).

**Memory comparison:**
```
Parameters alone (inference): 100%
+ Adam optimizer state:       +200% (m + v)
+ Forward activations:        + varies with B and depth
+ Gradients:                  +100%
TOTAL training overhead:      4-6× over inference
```

```bash
./build/04_training_loop 256 500    # batch=256, 500 steps
./build/04_training_loop 512 1000   # larger batch, more steps
```

---

## What PyTorch maps to

| CUDA code               | PyTorch equivalent                          |
|------------------------|---------------------------------------------|
| `cublasSgemm`          | `F.linear` → dispatches to cuBLAS           |
| `biasReluFwd` kernel   | `F.relu(F.linear(x, W, b))`                |
| `reluBwdKernel`        | `torch.ops.aten.threshold_backward`         |
| `biasGrad` kernel      | `torch.ops.aten.sum(dim=0)`                 |
| `adamUpdateKernel`     | `torch.optim.Adam.step`                     |
| `cudaMemset(grad, 0)`  | `optimizer.zero_grad()`                     |

`torch.compile()` identifies sequences of memory-bound elementwise kernels
(bias add, ReLU, dropout) and fuses them automatically — the same reason our
`biasReluFwd` is faster than separate bias and ReLU kernels.

---

## Exercises

**Exercise 1 (activations):** Add a Swish/SiLU kernel (`x * sigmoid(x)`) and
its backward. Used in LLaMA's MLP layers. How does its arithmetic intensity
compare to ReLU and GELU? Run Nsight Compute roofline to verify.

**Exercise 2 (forward):** Add a third layer to the MLP in `02_mlp_forward.cu`.
Measure how memory usage scales. At what depth does activation memory exceed
weight memory for B=512?

**Exercise 3 (backprop):** Implement gradient clipping in `03_backprop.cu`:
compute the global gradient norm across all parameter gradients, then scale
all gradients by `min(1, max_norm / global_norm)`. This requires a parallel
reduction across all parameter arrays.

**Exercise 4 (training):** Replace SGD/Adam with Adagrad or RMSProp in
`04_training_loop.cu`. Each has a different update rule but the same kernel
structure. Does the model converge faster or slower on the synthetic task?

**Exercise 5 (memory):** Implement gradient checkpointing for `03_backprop.cu`:
don't save `h_pre` and `h` during the forward pass. Instead, recompute them
during the backward pass when they're needed. How much memory does this save?
What is the compute overhead?

---

## License

MIT.
