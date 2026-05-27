#!/usr/bin/env bash
# scripts/run_all.sh — build and run every Post 2 benchmark in sequence
# Usage: bash scripts/run_all.sh [GPU_ARCH]
#   GPU_ARCH defaults to 'native' (auto-detect). Override with sm_80, sm_90, etc.

set -euo pipefail

ARCH="${1:-native}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/build"

echo "═══════════════════════════════════════════════════════"
echo "  Scaling GPUs — Post 2: Your First CUDA Kernels"
echo "  Building for ARCH=$ARCH"
echo "═══════════════════════════════════════════════════════"
echo ""

cd "$ROOT"
make clean > /dev/null 2>&1 || true
make all ARCH="$ARCH"

echo ""
echo "── 1. Vector addition (n = 2^24 = 16 M elements) ──────"
"$BUILD/vec_add"

echo ""
echo "── 2a. Matrix multiply  512 × 512  ────────────────────"
"$BUILD/matmul" 512

echo ""
echo "── 2b. Matrix multiply 1024 × 1024 ────────────────────"
"$BUILD/matmul" 1024

echo ""
echo "── 2c. Matrix multiply 2048 × 2048 ────────────────────"
"$BUILD/matmul" 2048

echo ""
echo "── 3. Occupancy explorer ───────────────────────────────"
"$BUILD/occupancy"

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  All benchmarks complete."
echo "  Next step: open a profile with Nsight Systems:"
echo "    make profile-matmul"
echo "═══════════════════════════════════════════════════════"
