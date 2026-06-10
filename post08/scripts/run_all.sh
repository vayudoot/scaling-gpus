#!/usr/bin/env bash
# scripts/run_all.sh -- Post 8: Multi-GPU Infrastructure
set -euo pipefail
ARCH="${1:-native}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
make clean > /dev/null 2>&1 || true
make all ARCH="$ARCH"

sep() { printf '\n=======================================================\n'; }

sep; echo " 1. P2P / NVLink bandwidth (or H2D/D2H on single GPU)"
./build/01_p2p_bandwidth

sep; echo " 2. Ring-AllReduce from scratch"
./build/02_ring_allreduce

sep; echo " 3. NCCL collectives (or install instructions if absent)"
./build/03_nccl_collectives

sep; echo " 4. Communication cost model"
./build/04_comm_cost_model

sep
echo " Done."
echo " On a single GPU: programs 1-2 simulate, 3 needs NCCL, 4 is pure analysis."
echo " On a multi-GPU node: all four show real cross-GPU behaviour."
sep
