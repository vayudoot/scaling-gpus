#!/usr/bin/env bash
# scripts/run_all.sh — Post 3: GPU Memory: The Real Bottleneck
set -euo pipefail
ARCH="${1:-native}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
make clean > /dev/null 2>&1 || true
make all ARCH="$ARCH"

echo ""
echo "═══════════════════════════════════════════════════════"
echo " 1. Memory coalescing — stride 1 vs strided vs random"
echo "═══════════════════════════════════════════════════════"
./build/01_coalescing

echo ""
echo "═══════════════════════════════════════════════════════"
echo " 2. Matrix transpose — naive / smem / smem+pad"
echo "═══════════════════════════════════════════════════════"
./build/02_transpose 4096

echo ""
echo "═══════════════════════════════════════════════════════"
echo " 3. Shared memory bank conflicts — isolated benchmark"
echo "═══════════════════════════════════════════════════════"
./build/03_bank_conflicts

echo ""
echo "═══════════════════════════════════════════════════════"
echo " 4. Kernel fusion — unfused vs fused scale+LN+ReLU"
echo "═══════════════════════════════════════════════════════"
./build/04_fusion

echo ""
echo "═══════════════════════════════════════════════════════"
echo " 5. Roofline — measuring AI and regime for each kernel"
echo "═══════════════════════════════════════════════════════"
./build/05_roofline

echo ""
echo "═══════════════════════════════════════════════════════"
echo " All done. Next steps:"
echo "   make ncu-bank-conflicts   — confirm conflict counts with Nsight"
echo "   make ncu-roofline         — full roofline report"
echo "   make profile-transpose    — Nsight Systems timeline"
echo "═══════════════════════════════════════════════════════"
