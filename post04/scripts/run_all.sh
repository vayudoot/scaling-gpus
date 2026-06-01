#!/usr/bin/env bash
# scripts/run_all.sh — Post 4: CUDA Streams and Async Execution
set -euo pipefail
ARCH="${1:-native}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
make clean > /dev/null 2>&1 || true
make all ARCH="$ARCH"

sep() { printf '\n═══════════════════════════════════════════════════════\n'; }

sep; echo " 1. Default stream — serialisation vs separate streams"
./build/01_default_stream

sep; echo " 2. Async copy — pageable vs pinned memory"
./build/02_async_copy

sep; echo " 3. Double-buffer pipeline — 20 batches"
./build/03_double_buffer 20

sep; echo " 4. CUDA events — precision timing and cross-stream sync"
./build/04_events

sep; echo " 5. CUDA Graphs — launch overhead elimination"
./build/05_cuda_graphs

sep
echo " All done.  Key next steps:"
echo "   make profile-streams   → see stream serialisation in Nsight"
echo "   make profile-pipeline  → see copy+compute overlap with NVTX labels"
echo "   make profile-graphs    → compare eager vs graph kernel gaps"
sep
