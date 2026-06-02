#!/usr/bin/env bash
# scripts/run_all.sh — Post 6: Attention on One GPU
set -euo pipefail
ARCH="${1:-native}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
make clean > /dev/null 2>&1 || true
make all ARCH="$ARCH"

sep() { printf '\n=======================================================\n'; }

sep; echo " 1. Softmax: naive / stable / online / warp-level"
./build/01_softmax 4096 1024
echo "---"; echo " (warp-level test, N_COLS=32):"
./build/01_softmax 4096 32

sep; echo " 2. Naive attention: step breakdown + N^2 scaling"
./build/02_attention_naive

sep; echo " 3. FlashAttention: correctness + speedup comparison"
./build/03_flash_attention

sep; echo " 4. Multi-head attention pipeline"
./build/04_multihead_attention 2 512

sep
echo " All done. Suggested profiling:"
echo "   make ncu-flash            -> verify O(N) HBM traffic"
echo "   make ncu-naive-softmax    -> see O(N^2) HBM traffic"
echo "   make profile-flash        -> Nsight Systems timeline"
sep
