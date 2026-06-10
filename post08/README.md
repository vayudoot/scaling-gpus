# Scaling GPUs -- Post 8: Multi-GPU Infrastructure

Code for **Post 8** of the [Scaling GPUs](https://substack.com/TODO) series.
Covers GPU interconnects (NVLink, PCIe, InfiniBand), peer-to-peer access,
the ring-AllReduce algorithm, NCCL collectives, and the communication cost
model that determines when distributed training scales.

---

## Structure

```
scaling_gpus_post08/
├── include/
│   └── utils.cuh               CUDA_CHECK, GpuTimer, device enumeration
├── src/
│   ├── 01_p2p_bandwidth.cu     P2P access matrix + NVLink/PCIe bandwidth
│   ├── 02_ring_allreduce.cu    Ring-AllReduce from scratch (real or simulated)
│   ├── 03_nccl_collectives.cu  NCCL AllReduce/Broadcast/AllGather/ReduceScatter
│   └── 04_comm_cost_model.cu   Alpha-beta model, scaling analysis (no GPU needed)
├── scripts/
│   └── run_all.sh
└── Makefile
```

**Runs on a single GPU.** Most readers have one GPU. Programs adapt:
- `01`: measures H2D/D2H bandwidth instead of P2P
- `02`: simulates the ring algorithm using separate buffers as virtual GPUs
- `03`: runs with a single-GPU NCCL communicator (or prints install steps if NCCL absent)
- `04`: pure analysis, no GPU traffic at all

On a 2+ GPU node, programs `01`-`03` show real cross-GPU behaviour.

```bash
make ARCH=sm_89 && make run
```

---

## Programs

### 1. `01_p2p_bandwidth` -- The interconnect hierarchy

Queries the peer-to-peer access matrix and measures bandwidth for each path:

```
       GPU0  GPU1  GPU2  GPU3
  GPU0  --    Y     Y     Y
  GPU1  Y     --    Y     Y
  ...
```

Then measures actual transfer bandwidth and classifies each link:
```
>200 GB/s   -> NVLink   (excellent for multi-GPU training)
25-60 GB/s  -> PCIe P2P (acceptable for small models)
<25 GB/s    -> host-staged (communication will bottleneck)
```

**Single-GPU fallback:** measures host-to-device and device-to-host bandwidth
over PCIe, which is the relevant number for data loading.

Key APIs: `cudaDeviceCanAccessPeer`, `cudaDeviceEnablePeerAccess`,
`cudaDeviceGetP2PAttribute`, `cudaMemcpyPeer`.

### 2. `02_ring_allreduce` -- The algorithm behind data-parallel training

Implements ring-AllReduce from scratch. AllReduce sums arrays across all GPUs
so every GPU ends with the total -- exactly what gradient averaging needs.

**The two phases:**
```
Phase 1 (reduce-scatter): P-1 steps. Chunks circulate around the ring,
  each GPU accumulating the chunks that pass through it. After P-1 steps,
  GPU r holds the complete sum for exactly one chunk.

Phase 2 (all-gather): P-1 steps. The completed chunks circulate so every
  GPU ends up with every completed chunk.
```

**Why it scales:** data moved per GPU is `2*(P-1)/P * N`, which approaches `2N`
as P grows -- independent of the number of GPUs:

```
P      Naive bytes/GPU   Ring bytes/GPU   Ring wins
2          4.2 MB            4.2 MB          1.0x
8         29.4 MB            7.3 MB          4.0x
64       264.2 MB            8.3 MB         32.0x
256        1.0 GB            8.4 MB        128.0x
```

Naive AllReduce (every GPU sends to every GPU) grows linearly with P. Ring is
constant. This is why all frameworks use ring or tree algorithms.

**Single-GPU simulation:** models P virtual GPUs as P device buffers, runs the
real algorithm with D2D copies, and verifies the sum is correct. The algorithm
is identical; only the physical links differ.

### 3. `03_nccl_collectives` -- What PyTorch actually calls

NCCL is the production library. It implements ring and tree algorithms,
auto-detects topology, and tunes to the hardware. This program demonstrates
the four collectives that matter for ML:

| Collective     | What it does                          | Used for                    |
|----------------|---------------------------------------|-----------------------------|
| AllReduce      | sum across GPUs, result on all        | data-parallel gradients     |
| Broadcast      | one GPU's data to all                 | initial weight distribution |
| AllGather      | concat each GPU's shard onto all      | tensor-parallel outputs     |
| ReduceScatter  | sum then shard across GPUs            | ZeRO, tensor-parallel       |

Note: `AllReduce = ReduceScatter + AllGather`. That decomposition *is* the
ring algorithm from program 2.

**NCCL is optional.** If `<nccl.h>` isn't installed, the program compiles to a
stub explaining how to install it. The Makefile auto-detects NCCL:
```bash
sudo apt install libnccl2 libnccl-dev    # Ubuntu
conda install -c conda-forge nccl         # conda
```

### 4. `04_comm_cost_model` -- When does scaling break?

Pure analysis tool (no GPU traffic). Computes the communication cost using the
alpha-beta model and answers the central question: does compute per step exceed
communication per step?

**The alpha-beta model for a ring collective:**
```
T = alpha * 2*(P-1)  +  beta * 2*(P-1)/P * N
    \_____________/      \__________________/
     latency term         bandwidth term
```
- Small messages (biases, layer norms): latency-bound -> NCCL fuses them into buckets
- Large messages (weight matrices): bandwidth-bound -> need fast links

**AllReduce time for a 7B model gradient (1.4 GB BF16) across 8 GPUs:**
```
NVLink 4 (H100)   900 GB/s    ~3 ms
NVLink 3 (A100)   600 GB/s    ~4 ms
PCIe 5.0 x16       64 GB/s    ~38 ms
InfiniBand HDR     25 GB/s    ~98 ms
Ethernet 100GbE    12.5 GB/s  ~196 ms
```

**Compute/communication ratio:** shows that larger batch-per-GPU hides
communication better. Below a threshold batch size, communication dominates
and adding GPUs stops helping -- the practical limit of data parallelism.

---

## The communication hierarchy (the key mental model)

```
Within a GPU  : HBM         ~3 TB/s     (post 3)
Within a node : NVLink      ~900 GB/s   (this post)
Between nodes : InfiniBand  ~25-50 GB/s (this post)
```

Each tier is 10-100x slower than the one above. Good distributed design keeps
the most frequent communication on the fastest tier:
- Tensor parallel (communicates every layer) -> stays on NVLink within a node
- Pipeline parallel (communicates at layer boundaries) -> tolerates InfiniBand
- Data parallel (communicates once per step) -> tolerates the slowest links

Posts 9 and 10 build on this to cover tensor, pipeline, and 3D parallelism.

---

## Exercises

**Exercise 1 (bandwidth):** On a multi-GPU node, run `01_p2p_bandwidth` and
compare the measured NVLink bandwidth to the spec sheet for your GPU. What
fraction of peak do you achieve? Why is it less than 100%?

**Exercise 2 (ring):** Modify `02_ring_allreduce.cu` to implement AllReduce as
a tree (binary-tree reduce to root, then broadcast back). For what P and message
sizes does tree beat ring? (Hint: tree has lower latency, ring has lower
bandwidth cost.)

**Exercise 3 (ring):** The current ring sends one chunk per step. Implement a
pipelined version that splits each chunk into sub-chunks and overlaps the send
of sub-chunk i+1 with the reduce of sub-chunk i. Measure the improvement.

**Exercise 4 (NCCL):** On a multi-GPU node, use `ncclCommInitRank` with one
process per GPU (instead of `ncclCommInitAll` single-process). This is how
PyTorch DDP actually works. Launch with MPI or `torchrun`.

**Exercise 5 (cost model):** Extend `04_comm_cost_model.cu` to model pipeline
parallelism's bubble overhead. For a given number of pipeline stages and
micro-batches, what fraction of time is wasted in the pipeline fill/drain?
(This is covered in post 10.)

---

## License

MIT.
