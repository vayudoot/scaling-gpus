#!/usr/bin/env bash
# scripts/run_all.sh -- Post 9: Data and Tensor Parallelism
set -euo pipefail
ARCH="${1:-native}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
make clean > /dev/null 2>&1 || true
make all ARCH="$ARCH"

sep() { printf '\n=======================================================\n'; }

sep; echo " 1. Data parallel (DDP): gradient averaging equivalence"
./build/01_data_parallel 4 256

sep; echo " 2. ZeRO: optimizer-state sharding memory savings"
./build/02_zero_sharding

sep; echo " 3. Tensor parallel: Megatron MLP (column + row parallel)"
./build/03_tensor_parallel 4

sep; echo " 4. 2D mesh: combining tensor and data parallelism"
./build/04_2d_mesh 4 4

sep
echo " Done. All results verified against single-GPU references."
echo " Post 10 adds pipeline parallelism and mixture-of-experts."
sep
