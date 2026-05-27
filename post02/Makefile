# Makefile — Scaling GPUs / Post 2: Your First CUDA Kernels
#
# Usage:
#   make           — build all targets
#   make vec_add   — build only vec_add
#   make run       — build and run all benchmarks
#   make clean     — remove build artefacts
#
# Compiler flags:
#   -O3            production optimisation
#   -arch=sm_80    target A100 (Ampere). Change to sm_89 (L4/RTX 4090),
#                  sm_90 (H100), or sm_70 (V100) to match your GPU.
#                  'native' auto-detects the installed GPU.
#   -lineinfo      embed source line info for profiler (no debug overhead)
#   -I include     add the include/ directory to the search path

NVCC      := nvcc
ARCH      := native   # auto-detect; replace with sm_80, sm_90, etc. if needed
CFLAGS    := -O3 -arch=$(ARCH) -lineinfo -I include
DBG_FLAGS := -g -G -O0 -arch=$(ARCH) -I include  # for cuda-gdb / compute-sanitizer

BUILD_DIR := build
SRC_DIR   := src

TARGETS := vec_add matmul occupancy

.PHONY: all run clean debug help

all: $(TARGETS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# ── Build rules ───────────────────────────────────────────────────────────────

vec_add: $(BUILD_DIR) $(SRC_DIR)/vec_add.cu include/timer.cuh
	$(NVCC) $(CFLAGS) $(SRC_DIR)/vec_add.cu -o $(BUILD_DIR)/vec_add

matmul: $(BUILD_DIR) $(SRC_DIR)/matmul.cu include/timer.cuh
	$(NVCC) $(CFLAGS) $(SRC_DIR)/matmul.cu -o $(BUILD_DIR)/matmul

occupancy: $(BUILD_DIR) $(SRC_DIR)/occupancy.cu include/timer.cuh
	$(NVCC) $(CFLAGS) $(SRC_DIR)/occupancy.cu -o $(BUILD_DIR)/occupancy

# Debug builds (for compute-sanitizer and cuda-gdb)
debug: $(BUILD_DIR)
	$(NVCC) $(DBG_FLAGS) $(SRC_DIR)/vec_add.cu  -o $(BUILD_DIR)/vec_add_dbg
	$(NVCC) $(DBG_FLAGS) $(SRC_DIR)/matmul.cu   -o $(BUILD_DIR)/matmul_dbg
	$(NVCC) $(DBG_FLAGS) $(SRC_DIR)/occupancy.cu -o $(BUILD_DIR)/occupancy_dbg

# ── Run all benchmarks ────────────────────────────────────────────────────────

run: all
	@echo ""
	@echo "═══════════════════════════════════════════"
	@echo " vec_add  (16M elements)"
	@echo "═══════════════════════════════════════════"
	./$(BUILD_DIR)/vec_add

	@echo ""
	@echo "═══════════════════════════════════════════"
	@echo " matmul  (1024 × 1024)"
	@echo "═══════════════════════════════════════════"
	./$(BUILD_DIR)/matmul 1024

	@echo ""
	@echo "═══════════════════════════════════════════"
	@echo " occupancy explorer"
	@echo "═══════════════════════════════════════════"
	./$(BUILD_DIR)/occupancy

# ── Profiling shortcuts ───────────────────────────────────────────────────────

profile-vec:
	nsys profile --trace=cuda --output=profiles/vec_add ./$(BUILD_DIR)/vec_add

profile-matmul:
	nsys profile --trace=cuda --output=profiles/matmul ./$(BUILD_DIR)/matmul 2048

ncu-matmul-naive:
	ncu --kernel-name matmulNaive --set full ./$(BUILD_DIR)/matmul 512

ncu-matmul-tiled:
	ncu --kernel-name matmulTiled --set full ./$(BUILD_DIR)/matmul 512

sanitize-vec:
	compute-sanitizer --tool memcheck ./$(BUILD_DIR)/vec_add_dbg

sanitize-matmul:
	compute-sanitizer --tool memcheck ./$(BUILD_DIR)/matmul_dbg 256

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
	rm -rf $(BUILD_DIR) profiles

# ── Help ──────────────────────────────────────────────────────────────────────

help:
	@echo "Targets:"
	@echo "  all              Build all programs (default)"
	@echo "  vec_add          Vector addition benchmark"
	@echo "  matmul           Naïve vs tiled matrix multiply"
	@echo "  occupancy        Block-size vs occupancy explorer"
	@echo "  debug            Debug builds (-g -G -O0)"
	@echo "  run              Build and run all benchmarks"
	@echo "  profile-vec      Nsight Systems profile of vec_add"
	@echo "  profile-matmul   Nsight Systems profile of matmul"
	@echo "  ncu-matmul-naive Nsight Compute on naïve matmul kernel"
	@echo "  ncu-matmul-tiled Nsight Compute on tiled matmul kernel"
	@echo "  sanitize-vec     compute-sanitizer memcheck on vec_add"
	@echo "  sanitize-matmul  compute-sanitizer memcheck on matmul"
	@echo "  clean            Remove build artefacts"
	@echo ""
	@echo "GPU architecture (ARCH=):"
	@echo "  native  auto-detect (default)"
	@echo "  sm_70   V100"
	@echo "  sm_80   A100"
	@echo "  sm_89   L4, RTX 4090"
	@echo "  sm_90   H100"
	@echo ""
	@echo "Example: make ARCH=sm_80 run"
