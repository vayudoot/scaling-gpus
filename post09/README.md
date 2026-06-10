# Scaling GPUs -- Post 9: Data and Tensor Parallelism

Code for **Post 9** of the [Scaling GPUs](https://substack.com/TODO) series.
Covers data parallelism (DDP), ZeRO optimizer-state sharding, tensor parallelism
(Megatron-style column/row split), and combining them in a 2D device mesh.

---

## Structure

```
scaling_gpus_post09/
├── include/
│   └── utils.cuh              CUDA_CHECK, CUBLAS_CHECK, GpuTimer
├── src/
│   ├── 01_data_parallel.cu    DDP gradient averaging + equivalence proof
│   ├── 02_zero_sharding.cu    ZeRO stages 1/2/3 memory math + simulation
│   ├── 03_tensor_parallel.cu  Megatron MLP: column + row parallel linears
│   └── 04_2d_mesh.cu          TP x DP mesh, communication analysis
├── scripts/
│   └── run_all.sh
└── Makefile
```

**Runs on a single GPU.** Every program simulates P GPUs using separate
buffers and **verifies the parallel result against a single-GPU reference**,
so you can confirm the math is exact even without a GPU cluster.

```bash
make ARCH=sm_89 && make run
```

---

## Programs

### 1. `01_data_parallel` -- The DDP equivalence proof

Data parallelism: every GPU has a full model copy, processes a batch shard,
and the gradients are AllReduced (summed and averaged). The defining property:

> **DDP on P GPUs with batch B/P each produces the SAME gradient as one GPU
> processing the full batch B.**

This program computes both and verifies they match to within 1e-4. That
equivalence is *why* DDP is correct -- it's not an approximation.

**The key limitation it exposes:** DDP does not reduce per-GPU memory. Every
GPU still stores the full model, gradients, and optimizer state. A 7B model
under Adam needs ~112 GB per GPU regardless of how many GPUs you add. That's
what ZeRO fixes.

### 2. `02_zero_sharding` -- Sharding the optimizer state

ZeRO eliminates the memory redundancy of DDP by sharding training state across
the data-parallel GPUs in three progressive stages:

```
Per-GPU memory for a 70B model (P=8 GPUs, mixed-precision Adam):

Model    DDP        ZeRO-1     ZeRO-2     ZeRO-3
70B      1120 GB    637 GB     623 GB     140 GB
```

- **Stage 1**: shard optimizer state (FP32 master, Adam m & v) -> `/P`
- **Stage 2**: also shard gradients -> `/P`
- **Stage 3** (FSDP): also shard parameters -> everything `/P`

The program simulates Stage 1 and **verifies the sharded Adam update produces
exactly the same parameters as an unsharded optimizer** -- each GPU updates
its own slice of params using its slice of the state.

**The trade-off:** Stages 1 and 2 add no communication beyond DDP's existing
AllReduce (it just becomes a reduce-scatter + all-gather). Stage 3 all-gathers
parameters before every layer, so it needs a fast interconnect.

### 3. `03_tensor_parallel` -- The Megatron MLP (the centerpiece)

Tensor parallelism splits a single layer's weight matrix across GPUs, so a
layer too large for one GPU can still run. There are two ways to split a
linear layer `Y = X @ W`:

**Column-parallel** (split the output dimension):
```
Each GPU computes Y_r = X @ A_r independently.  NO communication.
Output is split across GPUs by column.
```

**Row-parallel** (split the input dimension):
```
Each GPU computes a PARTIAL sum Y_r = X_r @ B_r.
The partials are summed via ONE AllReduce.
```

**The Megatron MLP trick** -- `Y = GeLU(X @ A) @ B`:
- `A` is column-parallel (splits the hidden dim, **zero communication**)
- `B` is row-parallel (consumes the split, **one AllReduce**)

The whole MLP block needs only **one AllReduce** in the forward pass because
the column-split output of the first linear feeds directly into the row-split
second linear. Reversing the order would force an extra AllGather between them.

The program implements both splits, composes the full MLP, and **verifies the
tensor-parallel output equals the single-GPU CPU reference** to within 1e-3.

**Why TP stays within a node:** it communicates inside *every layer* (64 times
per step for a 32-layer model). That frequency demands NVLink. Crossing nodes
over InfiniBand would make the per-layer AllReduce dominate. This is why TP
degree usually equals GPUs-per-node (8).

### 4. `04_2d_mesh` -- Combining TP and DP

Real training combines strategies. The simplest combination is a 2D mesh:
tensor-parallel within a node, data-parallel across nodes.

```
        TP axis (NVLink) ->
      +----+----+----+----+
   DP | G0 | G1 | G2 | G3 |  <- TP group (shares a layer)
   |  +----+----+----+----+
   v  | G4 | G5 | G6 | G7 |
      +----+----+----+----+
```

- Each **row** is a tensor-parallel group (fast, frequent, within-node comm)
- Each **column** is a data-parallel group (slow, infrequent, cross-node comm)

The program builds the mesh, maps each GPU's `(tp_rank, dp_rank)` coordinates,
and analyses the communication volume on each axis:

```
TP axis: 2 AllReduces per layer x 32 layers = frequent  -> NVLink
DP axis: 1 gradient AllReduce per step       = infrequent -> InfiniBand
```

It also computes the optimal TP/DP split for a fixed GPU budget, showing how
to pick the smallest TP that fits the model in memory while minimizing the
expensive per-layer TP communication.

---

## The parallelism decision tree

```
Model fits on one GPU?
├── Yes -> Data Parallel (DDP)
│          Memory tight? -> add ZeRO-1/2 (nearly free)
│                           or ZeRO-3/FSDP (more comm, max savings)
└── No  -> must split the model
           ├── Individual layer too big -> Tensor Parallel (within node)
           └── Too many layers          -> Pipeline Parallel (post 10)
                                           usually combined: TP x PP x DP
```

---

## Exercises

**Exercise 1 (DDP):** Modify `01_data_parallel.cu` so the per-GPU batches have
different sizes (uneven sharding). What scaling factor keeps the averaged
gradient correct? (Hint: weight each shard by its sample count.)

**Exercise 2 (ZeRO):** Extend `02_zero_sharding.cu` to simulate ZeRO-3:
parameters are sharded, so before the forward pass each GPU must all-gather the
full parameter tensor. Measure the extra memory traffic vs the memory saved.

**Exercise 3 (TP):** Add the backward pass to `03_tensor_parallel.cu`. The
backward of a column-parallel layer needs an AllReduce; the backward of a
row-parallel layer needs none. Verify the gradient against a single-GPU
reference. (This is the symmetric dual of the forward pass.)

**Exercise 4 (TP):** Implement tensor-parallel attention: split the attention
heads across GPUs (each GPU computes a subset of heads), then AllReduce the
output projection. How does the communication compare to the MLP?

**Exercise 5 (mesh):** Extend `04_2d_mesh.cu` to compute the end-to-end step
time given per-axis link bandwidths. Find the TP/DP split that minimizes step
time (not just memory) for a 70B model on 64 H100s.

---

## License

MIT.
