# Makefile -- Scaling GPUs / Post 11: Profiling and Optimization

NVCC    := nvcc
ARCH    := native
CFLAGS  := -O3 -arch=$(ARCH) -lineinfo -I include --std=c++17
DBG     := -g -G -O0 -arch=$(ARCH) -I include --std=c++17
BUILD   := build
LIBS    := -lcublas

TARGETS := 01_pathological_step 02_benchmark_pitfalls 03_ncu_target

.PHONY: all run debug clean help profiles

all: $(TARGETS)

$(BUILD):
	mkdir -p $(BUILD)

01_pathological_step: $(BUILD) src/01_pathological_step.cu include/utils.cuh
	$(NVCC) $(CFLAGS) src/01_pathological_step.cu $(LIBS) -o $(BUILD)/01_pathological_step

02_benchmark_pitfalls: $(BUILD) src/02_benchmark_pitfalls.cu include/utils.cuh
	$(NVCC) $(CFLAGS) src/02_benchmark_pitfalls.cu $(LIBS) -o $(BUILD)/02_benchmark_pitfalls

03_ncu_target: $(BUILD) src/03_ncu_target.cu include/utils.cuh
	$(NVCC) $(CFLAGS) src/03_ncu_target.cu -o $(BUILD)/03_ncu_target

debug: $(BUILD)
	$(NVCC) $(DBG) src/01_pathological_step.cu  $(LIBS) -o $(BUILD)/01_pathological_step_dbg
	$(NVCC) $(DBG) src/02_benchmark_pitfalls.cu $(LIBS) -o $(BUILD)/02_benchmark_pitfalls_dbg
	$(NVCC) $(DBG) src/03_ncu_target.cu         -o $(BUILD)/03_ncu_target_dbg

run: all
	@echo ""
	@echo "=== 01: Pathological step (baseline) ==="
	./$(BUILD)/01_pathological_step
	@echo ""
	@echo "=== 01: Pathological step (--fixed) ==="
	./$(BUILD)/01_pathological_step --fixed
	@echo ""
	@echo "=== 02: Benchmarking pitfalls ==="
	./$(BUILD)/02_benchmark_pitfalls
	@echo ""
	@echo "=== 03: ncu target (three reductions) ==="
	./$(BUILD)/03_ncu_target

# The whole point of this post: profile targets ready to go.
profiles: all
	mkdir -p profiles
	nsys profile -t cuda,nvtx --output=profiles/step_slow ./$(BUILD)/01_pathological_step
	nsys profile -t cuda,nvtx --output=profiles/step_fast ./$(BUILD)/01_pathological_step --fixed
	@echo "Open profiles/step_slow.nsys-rep and step_fast.nsys-rep side by side."

ncu-atomic:
	ncu --kernel-name atomicAllKernel --set full ./$(BUILD)/03_ncu_target

ncu-strided:
	ncu --kernel-name stridedKernel --set full ./$(BUILD)/03_ncu_target

ncu-coalesced:
	ncu --kernel-name coalescedKernel --set full ./$(BUILD)/03_ncu_target

clean:
	rm -rf $(BUILD) profiles

help:
	@echo "Targets:"
	@echo "  all              Build all three programs"
	@echo "  run              Build and run all (01 in both modes)"
	@echo "  profiles         nsys capture of 01 in both modes (needs nsys)"
	@echo "  ncu-atomic       ncu deep-dive on reduction variant A (needs ncu)"
	@echo "  ncu-strided      ncu deep-dive on variant B"
	@echo "  ncu-coalesced    ncu deep-dive on variant C"
	@echo "  clean            Remove build artefacts and profiles"
	@echo ""
	@echo "NVTX is optional (header-only): ranges appear if nvtx3 headers exist."
	@echo "Architecture: make ARCH=sm_80 (A100) / sm_89 (L4/4090) / sm_90 (H100)"
