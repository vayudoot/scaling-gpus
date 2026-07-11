#!/usr/bin/env bash
# scripts/run_all.sh -- Post 10: Pipeline Parallelism and MoE
set -euo pipefail
ARCH="${1:-native}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
make clean > /dev/null 2>&1 || true
make all ARCH="$ARCH"

sep() { printf '\n=======================================================\n'; }

sep; echo " 1. Pipeline schedules: GPipe vs 1F1B (simulation)"
./build/01_pipeline_schedule 4 6

sep; echo " 2. Real pipelined execution: streams as virtual stages"
./build/02_pipeline_parallel 4

sep; echo " 3. MoE layer: routing, expert GEMMs, imbalance, capacity"
./build/03_moe_layer

sep; echo " 4. AllToAll simulation + expert-parallel cost model"
./build/04_alltoall_cost

sep
echo " Done. All parallel results verified against references."
echo " Post 11 covers profiling the schedules you just built."
sep
