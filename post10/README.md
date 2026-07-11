# Scaling GPUs -- Post 10: Pipeline Parallelism and Mixture-of-Experts

Code for **Post 10** of the [Scaling GPUs](https://substack.com/TODO) series.
Covers pipeline schedules (GPipe, 1F1B, the bubble), a real pipelined
execution built from CUDA streams and events, a complete MoE layer with
top-1 routing, and the AllToAll communication that expert parallelism rides on.

---

## Structure

```
scaling_gpus_post10/
├── include/
│   └── utils.cuh                CUDA_CHECK, CUBLAS_CHECK, GpuTimer
├── src/
│   ├── 01_pipeline_schedule.cu  GPipe vs 1F1B simulator, bubble math
│   ├── 02_pipeline_parallel.cu  real pipeline: streams + events, verified
│   ├── 03_moe_layer.cu          router -> group -> expert FFNs -> ungroup
│   └── 04_alltoall_cost.cu      AllToAll simulation + EP cost model
├── scripts/
│   └── run_all.sh
└── Makefile
```

**Runs on a single GPU** (program 01 needs no GPU at all). As throughout the
series, every parallel decomposition is **verified against a reference**.

```bash
make ARCH=sm_89 && make run
```

---

## Programs

### 1. `01_pipeline_schedule` -- GPipe vs 1F1B, simulated exactly

An event-driven simulator that enforces the true pipeline dependencies
(`F_i@k` needs `F_i@k-1`; `B_i@k` needs `B_i@k+1`) and the 1F1B policy
(prefer the oldest ready backward; cap in-flight micro-batches at `P-k`).
It prints ASCII Gantt charts for both schedules:

```
  stage 0 0123....a4b5c.d.e.
  stage 1 .012.a3b4c5d.e...
  ...
```

**The result most people get wrong:** with equal-cost ops, GPipe and 1F1B
have the *same* makespan and the *same* bubble fraction:

```
bubble = (P-1) / (M + P-1)
```

1F1B's win is **memory**: GPipe holds M activation sets per stage; 1F1B caps
it at `P-k`, independent of M. That cap is what lets you raise M (to shrink
the bubble) without exhausting HBM. The simulator measures peak in-flight
activations for both schedules and prints the trade-off table, ending at the
rule of thumb `M >= 4P` for a sub-20% bubble.

### 2. `02_pipeline_parallel` -- a real pipeline from streams and events

Runs an 8-layer MLP split into P=4 stages on **one GPU**, using a CUDA stream
per stage and events as the stage-to-stage handoff:

```c
cudaStreamWaitEvent(stream[k], done[k-1][i], 0);   // wait for upstream
run_stage(k, i, ...);                              // this stage's layers
cudaEventRecord(done[k][i], stream[k]);            // release downstream
```

This is exactly the dependency structure a multi-GPU pipeline framework
builds -- there, the event/wait pair becomes a send/recv between GPUs, but
the schedule is identical. Two things are demonstrated:

1. **Correctness:** the pipelined output matches the sequential full-batch
   reference (verified to 1e-4).
2. **The schedule is real:** every (stage, micro-batch) op records begin/end
   events, and the program prints the measured Gantt -- you can see stage 1
   working on micro-batch 0 while stage 0 is already on micro-batch 1, and
   the fill staircase from program 01 appears in real timestamps.

On one GPU the stages share SMs, so expect similar total time, not speedup --
the point is the mechanics, which transfer unchanged to real multi-GPU PP.

### 3. `03_moe_layer` -- a complete Mixture-of-Experts layer

The full top-1 MoE forward, each step its own kernel or GEMM:

| Step | Implementation |
|------|----------------|
| Router | `logits = X @ Wr` (cuBLAS), argmax + softmax gate kernel |
| Group  | histogram (atomicAdd), exclusive scan, scatter kernel |
| Experts | one cuBLAS GEMM pair per expert on its contiguous token slice |
| Ungroup | gather kernel, scaled by the gate value |

The grouping permutation is the heart of it: **locally it's a scatter; across
GPUs it becomes the AllToAll**. Verified per-token against a CPU reference.

Then the two production headaches, measured on the actual batch:

- **Load imbalance:** the tokens-per-expert histogram (ASCII bars) and the
  imbalance factor -- with expert parallelism, the step runs at the speed of
  the hottest expert.
- **Capacity factor:** tokens/expert capped at `cf * N/E`; the table shows
  dropped-token percentage at cf = 1.0 / 1.25 / 1.5 / 2.0. Fixed capacity
  also makes AllToAll message sizes static -- what the comm layer wants.

Closes with the sparsity ledger: E× the parameters at ~1× the per-token
FLOPs, paid for in memory and communication.

### 4. `04_alltoall_cost` -- the collective MoE actually rides on

Simulates an AllToAll across P virtual GPUs with **uneven per-pair message
sizes** drawn from a skewed routing matrix, verifies every (src, dst) block
lands correctly, and then quantifies the two costs:

- **The straggler effect:** the busiest inbound lane sets the pace -- the
  imbalance from program 03, now visible as traffic.
- **Alpha-beta cost at scale:** per-layer AllToAll time vs EP degree on
  NVLink (900 GB/s) vs InfiniBand (50 GB/s). Two AllToAlls per MoE layer,
  times num_layers per step -- on IB the collective quickly rivals the
  expert compute, which is why EP stays inside the NVLink domain.

Ends with the 4D parallelism map (DP × TP × PP × EP): what each axis splits,
what it communicates, and where it should live.

---

## The one-sentence takeaways

- Pipeline bubble: `(P-1)/(M+P-1)` -- shrink it with micro-batches, and use
  1F1B so the micro-batches don't blow up activation memory.
- MoE: E× parameters at 1× FLOPs per token; the bill arrives as memory,
  load imbalance, and two AllToAlls per layer.

---

## Exercises

**Exercise 1 (schedule):** Extend `01_pipeline_schedule.cu` so backward ops
cost 2 slots (realistic B ≈ 2F). Does the bubble formula still hold? Derive
the corrected expression and check it against the simulator.

**Exercise 2 (schedule):** Implement the interleaved (virtual-stage) schedule:
each GPU owns 2 non-contiguous stages (GPU0 has stages 0 and 4, ...). Measure
the bubble reduction vs plain 1F1B at equal M.

**Exercise 3 (pipeline):** Add the backward pass to `02_pipeline_parallel.cu`
using the 1F1B order from program 01, still with streams and events. Verify
gradients against a sequential reference.

**Exercise 4 (MoE):** Extend `03_moe_layer.cu` to top-2 routing: each token
goes to its two highest-scoring experts, outputs combined with normalized
gate weights. What happens to the tokens-per-expert histogram and the FLOPs?

**Exercise 5 (MoE):** Implement the capacity factor for real: cap each
expert's slice at `cf * N/E` in the scatter kernel, route dropped tokens
through an identity path, and verify the output still matches the reference
for surviving tokens.

**Exercise 6 (AllToAll):** In `04_alltoall_cost.cu`, overlap the AllToAll
with the previous layer's expert compute using separate streams. How much of
the collective can you hide?

---

## License

MIT.
