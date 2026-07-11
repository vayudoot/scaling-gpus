#!/usr/bin/env bash
# scripts/run_all.sh -- Post 11: Profiling and Optimization
set -euo pipefail
ARCH="${1:-native}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
make clean > /dev/null 2>&1 || true
make all ARCH="$ARCH"

sep() { printf '\n=======================================================\n'; }

sep; echo " 1a. Pathological training step (all four sins present)"
./build/01_pathological_step

sep; echo " 1b. Same step, --fixed (remedies applied)"
./build/01_pathological_step --fixed

sep; echo " 2. Benchmarking pitfalls: four wrong numbers vs the truth"
./build/02_benchmark_pitfalls

sep; echo " 3. ncu target: one reduction, three bottlenecks"
./build/03_ncu_target

sep
echo " Next: 'make profiles' captures nsys traces of 01 in both modes,"
echo " and 'make ncu-atomic|ncu-strided|ncu-coalesced' inspects program 03."
sep
