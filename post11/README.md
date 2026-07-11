# Scaling GPUs -- Post 11: Profiling and Optimization

Code for **Post 11** of the [Scaling GPUs](https://substack.com/TODO) series.
A profiling post needs something to profile: this repo provides the targets.
Three programs -- a deliberately pathological training step (with a `--fixed`
mode), the four benchmarking sins made executable, and a Nsight Compute
practice kernel set with known, distinct bottlenecks.

**Scope note:** the post's PyTorch Profiler and `torch.compile` sections are
Python-side and not duplicated here; CUDA Graphs (the fix for launch-bound
workloads) already has a full program in the **Post 4 repo**.

---

## Structure

```
scaling_gpus_post11/
├── include/
│   └── utils.cuh                 CUDA_CHECK, CUBLAS_CHECK, GpuTimer
├── src/
│   ├── 01_pathological_step.cu   the four timeline pathologies + --fixed
│   ├── 02_benchmark_pitfalls.cu  four wrong measurements vs the truth
│   └── 03_ncu_target.cu          one reduction, three bottlenecks, for ncu
├── scripts/
│   └── run_all.sh
└── Makefile                      includes nsys/ncu capture targets
```

```bash
make ARCH=sm_89 && make run
make profiles        # nsys captures of 01, both modes (needs nsys)
make ncu-strided     # ncu deep-dive on one variant of 03 (needs ncu)
```

NVTX is optional and header-only: ranges light up if `nvtx3` headers are
installed, and compile away to nothing otherwise.

---

## Programs

### 1. `01_pathological_step` -- the timeline you learn to read

A training-step-shaped workload containing all four pathologies from the
post's Nsight Systems section, each wrapped in an NVTX range so the trace
names them for you:

| # | Pathology | Built in as | The fix (`--fixed`) |
|---|-----------|-------------|---------------------|
| 1 | launch-bound | 200 tiny kernel launches | one fused kernel (at scale: CUDA Graphs, Post 4) |
| 2 | false serialization | two independent head GEMMs, one stream | two streams, event join |
| 3 | comm not hidden | simulated gradient AllReduce *after* compute | overlapped on a comm stream during forward (DDP-style: previous step's bucket) |
| 4 | sync readback | blocking 4-byte loss copy every step | async into pinned memory, polled every 10 steps |

Run both modes under nsys and diff the timelines:

```bash
nsys profile -t cuda,nvtx -o slow ./build/01_pathological_step
nsys profile -t cuda,nvtx -o fast ./build/01_pathological_step --fixed
```

Even without nsys, the program event-times every phase and prints the
before/after table -- the tiny-op chain typically collapses by 10-100x, and
`grad_comm` drops to near zero because it now runs *under* the forward pass.
The math is identical in both modes (same final loss): **the fixes change
the schedule, not the answer** -- the running theme of this series.

### 2. `02_benchmark_pitfalls` -- measuring wrong is worse than guessing

The same cuBLAS GEMM measured five ways. Four of them are the classic sins,
and the program prints exactly what each would have made you believe:

1. **Timing the first call** -- includes cuBLAS heuristics, module load, cold
   clocks; typically several times slower than steady state.
2. **CPU wall clock without a sync** -- kernel launches are asynchronous, so
   you time the *enqueue*: the program happily reports an absurd
   "N TFLOP/s" before showing the honest number.
3. **Reporting a single run** -- the min/median/max spread across 50 reps
   shows how far one unlucky sample lands.
4. **Including the H2D copy** -- real cost, wrong line item: that's PCIe,
   not compute.

Ends with the recipe used by every program in this series: warm up, event
pairs around device work only, median of many reps, verify the output.

### 3. `03_ncu_target` -- one reduction, three diagnoses

The same 64M-float sum implemented three ways, each with a known bottleneck,
all verified against a CPU reference:

| Variant | Trick | Bottleneck | ncu signature |
|---------|-------|------------|---------------|
| A `atomicAllKernel` | global atomicAdd per element | atomic serialization | low DRAM throughput, LG-throttle stalls |
| B `stridedKernel` | thread-contiguous chunks | **uncoalesced** loads (warp-strided) | ~32 sectors/request, long-scoreboard stalls |
| C `coalescedKernel` | grid-stride + shared tree | none -- honestly BW-bound | DRAM throughput near peak, ~4 sectors/request |

```bash
make ncu-atomic ncu-strided ncu-coalesced
```

The point: all three are "memory-bound" if you only look at achieved FLOPs,
but ncu tells you *which* memory problem you have -- serialization, access
pattern, or none. Variant B is the coalescing lesson from Post 3, now
visible in a profiler instead of a bandwidth table.

---

## The workflow this repo supports (from the post)

```
baseline MFU  ->  nsys (where does wall time go?)  ->  fix schedule issues
              ->  ncu on the hot kernel (why is IT slow?)  ->  fix the kernel
              ->  re-baseline. Repeat until the roofline says you're done.
```

Program 01 is the nsys half of that loop; program 03 is the ncu half;
program 02 keeps the numbers you collect honest.

---

## Exercises

**Exercise 1:** Capture `make profiles` and measure, in the GUI, the exact
gap lengths in the Compute row for the baseline. Which pathology costs the
most on *your* GPU? (The answer differs between a 4090 and an H100 -- launch
overhead is constant, compute is not.)

**Exercise 2:** Replace pathology 1's fused fix with a CUDA Graph capture of
the 200 tiny launches (see the Post 4 repo). Compare all three: 200 launches,
one fused kernel, one graph replay of 200 launches.

**Exercise 3:** In `02_benchmark_pitfalls.cu`, lock your GPU clocks
(`nvidia-smi -lgc`) and re-run. How much of the min/median/max spread was
clock drift?

**Exercise 4:** Add a variant D to `03_ncu_target.cu` that uses warp-shuffle
reduction (`__shfl_down_sync`, Post 6) instead of shared memory. What changes
in the ncu report -- and why is the bandwidth barely different?

**Exercise 5:** Take the MoE layer from the Post 10 repo, run it under nsys,
and find the gap between the per-expert GEMMs. Then batch the expert GEMMs
with `cublasGemmBatchedEx` and measure the difference.

---

## License

MIT.
