#!/usr/bin/env bash
# scripts/run_all.sh — Post 5: Building a Neural Network on One GPU
set -euo pipefail
ARCH="${1:-native}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
make clean > /dev/null 2>&1 || true
make all ARCH="$ARCH"

sep() { printf '\n=======================================================\n'; }

sep; echo " 1. Activation functions + fusion benchmark"
./build/01_activations

sep; echo " 2. MLP forward pass — tiled kernel vs cuBLAS"
./build/02_mlp_forward 512

sep; echo " 3. Backpropagation — every gradient step explicit"
./build/03_backprop 256

sep; echo " 4. Full training loop with Adam (500 steps)"
./build/04_training_loop 256 500

sep
echo " All done. Suggested next steps:"
echo "   make profile-train     -> Nsight Systems on training loop"
echo "   make ncu-activations   -> Nsight Compute roofline on GELU"
echo "   make ncu-adam          -> Verify Adam is memory-bound"
sep
