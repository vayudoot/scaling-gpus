# Makefile — Scaling GPUs / Post 3: GPU Memory: The Real Bottleneck

NVCC     := nvcc
ARCH     := native
CFLAGS   := -O3 -arch=$(ARCH) -lineinfo -I include --std=c++17
DBG      := -g -G -O0 -arch=$(ARCH) -I include --std=c++17
BUILD    := build

TARGETS  := 01_coalescing 02_transpose 03_bank_conflicts 04_fusion 05_roofline

.PHONY: all run debug clean profile-all sanitize-all help

all: $(TARGETS)

$(BUILD):
	mkdir -p $(BUILD)

01_coalescing: $(BUILD) src/01_coalescing.cu include/timer.cuh
	$(NVCC) $(CFLAGS) src/01_coalescing.cu -o $(BUILD)/01_coalescing

02_transpose: $(BUILD) src/02_transpose.cu include/timer.cuh
	$(NVCC) $(CFLAGS) src/02_transpose.cu -o $(BUILD)/02_transpose

03_bank_conflicts: $(BUILD) src/03_bank_conflicts.cu include/timer.cuh
	$(NVCC) $(CFLAGS) src/03_bank_conflicts.cu -o $(BUILD)/03_bank_conflicts

04_fusion: $(BUILD) src/04_fusion.cu include/timer.cuh
	$(NVCC) $(CFLAGS) src/04_fusion.cu -o $(BUILD)/04_fusion

05_roofline: $(BUILD) src/05_roofline.cu include/timer.cuh
	$(NVCC) $(CFLAGS) src/05_roofline.cu -o $(BUILD)/05_roofline

debug: $(BUILD)
	$(NVCC) $(DBG) src/01_coalescing.cu    -o $(BUILD)/01_coalescing_dbg
	$(NVCC) $(DBG) src/02_transpose.cu     -o $(BUILD)/02_transpose_dbg
	$(NVCC) $(DBG) src/03_bank_conflicts.cu -o $(BUILD)/03_bank_conflicts_dbg
	$(NVCC) $(DBG) src/04_fusion.cu        -o $(BUILD)/04_fusion_dbg
	$(NVCC) $(DBG) src/05_roofline.cu      -o $(BUILD)/05_roofline_dbg

# ── Run all ───────────────────────────────────────────────────────────────────
run: all
	@echo ""; @echo "═══ 01: Memory coalescing ═══"
	./$(BUILD)/01_coalescing
	@echo ""; @echo "═══ 02: Matrix transpose ═══"
	./$(BUILD)/02_transpose 4096
	@echo ""; @echo "═══ 03: Bank conflicts ═══"
	./$(BUILD)/03_bank_conflicts
	@echo ""; @echo "═══ 04: Kernel fusion ═══"
	./$(BUILD)/04_fusion
	@echo ""; @echo "═══ 05: Roofline measurement ═══"
	./$(BUILD)/05_roofline

# ── Nsight Systems profiles ───────────────────────────────────────────────────
profile-coalescing:
	mkdir -p profiles
	nsys profile --trace=cuda --output=profiles/coalescing ./$(BUILD)/01_coalescing

profile-transpose:
	mkdir -p profiles
	nsys profile --trace=cuda --output=profiles/transpose ./$(BUILD)/02_transpose 4096

profile-fusion:
	mkdir -p profiles
	nsys profile --trace=cuda --output=profiles/fusion ./$(BUILD)/04_fusion

# ── Nsight Compute — bank conflicts ──────────────────────────────────────────
ncu-bank-conflicts:
	ncu \
	  --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \
	  ./$(BUILD)/03_bank_conflicts

# ── Nsight Compute — roofline for all kernels ─────────────────────────────────
ncu-roofline:
	ncu --set full --output=profiles/roofline ./$(BUILD)/05_roofline

# ── Nsight Compute — transpose comparison ────────────────────────────────────
ncu-transpose-naive:
	ncu --kernel-name transposeNaive \
	    --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \
	    ./$(BUILD)/02_transpose 1024

ncu-transpose-padded:
	ncu --kernel-name transposeSmemPadded \
	    --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \
	    ./$(BUILD)/02_transpose 1024

# ── compute-sanitizer ─────────────────────────────────────────────────────────
sanitize-coalescing:
	compute-sanitizer --tool memcheck ./$(BUILD)/01_coalescing_dbg

sanitize-transpose:
	compute-sanitizer --tool memcheck  ./$(BUILD)/02_transpose_dbg 512
	compute-sanitizer --tool racecheck ./$(BUILD)/02_transpose_dbg 512

sanitize-bank:
	compute-sanitizer --tool racecheck ./$(BUILD)/03_bank_conflicts_dbg

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	rm -rf $(BUILD) profiles

help:
	@echo "Targets:"
	@echo "  all                 Build all five programs"
	@echo "  run                 Build and run all benchmarks"
	@echo "  debug               Debug builds (-g -G -O0)"
	@echo "  profile-coalescing  Nsight Systems on coalescing benchmark"
	@echo "  profile-transpose   Nsight Systems on transpose benchmark"
	@echo "  profile-fusion      Nsight Systems on fusion benchmark"
	@echo "  ncu-bank-conflicts  Nsight Compute — show bank conflict counts"
	@echo "  ncu-roofline        Nsight Compute — full roofline report"
	@echo "  ncu-transpose-naive   Nsight Compute — naive transpose kernel"
	@echo "  ncu-transpose-padded  Nsight Compute — padded transpose kernel"
	@echo "  sanitize-*          compute-sanitizer on individual programs"
	@echo "  clean               Remove build artefacts"
	@echo ""
	@echo "GPU architecture override:"
	@echo "  make ARCH=sm_80 all   (A100)"
	@echo "  make ARCH=sm_90 all   (H100)"
