#!/usr/bin/env bash
# scripts/run_all.sh -- Post 7: Mixed Precision and Quantization
set -euo pipefail
ARCH="${1:-native}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
make clean > /dev/null 2>&1 || true
make all ARCH="$ARCH"

sep() { printf '\n=======================================================\n'; }

sep; echo " 1. Precision formats: FP32 vs FP16 vs BF16"
./build/01_precision_formats

sep; echo " 2. Tensor Cores: WMMA API and the 15x gap"
./build/02_tensor_cores

sep; echo " 3. Mixed precision: FP32 master + FP16 forward"
./build/03_mixed_precision 256

sep; echo " 4. INT8 quantization: per-tensor vs per-channel, dp4a"
./build/04_quantization

sep; echo " 5. W4A16: 4-bit weights for LLM decode throughput"
./build/05_w4a16

sep
echo " Done. Key profiling commands:"
echo "   make ncu-tensor-core-util  -> verify Tensor Core utilization"
echo "   make ncu-w4a16             -> measure weight bytes loaded from HBM"
sep
