# Makefile -- Scaling GPUs / Post 9: Data and Tensor Parallelism

NVCC    := nvcc
ARCH    := native
CFLAGS  := -O3 -arch=$(ARCH) -lineinfo -I include --std=c++17
DBG     := -g -G -O0 -arch=$(ARCH) -I include --std=c++17
BUILD   := build
LIBS    := -lcublas

TARGETS := 01_data_parallel 02_zero_sharding 03_tensor_parallel 04_2d_mesh

.PHONY: all run debug clean help

all: $(TARGETS)

$(BUILD):
	mkdir -p $(BUILD)

01_data_parallel: $(BUILD) src/01_data_parallel.cu include/utils.cuh
	$(NVCC) $(CFLAGS) src/01_data_parallel.cu $(LIBS) -o $(BUILD)/01_data_parallel

02_zero_sharding: $(BUILD) src/02_zero_sharding.cu include/utils.cuh
	$(NVCC) $(CFLAGS) src/02_zero_sharding.cu $(LIBS) -o $(BUILD)/02_zero_sharding

03_tensor_parallel: $(BUILD) src/03_tensor_parallel.cu include/utils.cuh
	$(NVCC) $(CFLAGS) src/03_tensor_parallel.cu $(LIBS) -o $(BUILD)/03_tensor_parallel

04_2d_mesh: $(BUILD) src/04_2d_mesh.cu include/utils.cuh
	$(NVCC) $(CFLAGS) src/04_2d_mesh.cu $(LIBS) -o $(BUILD)/04_2d_mesh

debug: $(BUILD)
	$(NVCC) $(DBG) src/01_data_parallel.cu   $(LIBS) -o $(BUILD)/01_data_parallel_dbg
	$(NVCC) $(DBG) src/02_zero_sharding.cu   $(LIBS) -o $(BUILD)/02_zero_sharding_dbg
	$(NVCC) $(DBG) src/03_tensor_parallel.cu $(LIBS) -o $(BUILD)/03_tensor_parallel_dbg
	$(NVCC) $(DBG) src/04_2d_mesh.cu         $(LIBS) -o $(BUILD)/04_2d_mesh_dbg

run: all
	@echo ""; @echo "=== 01: Data parallel (DDP) gradient equivalence ==="
	./$(BUILD)/01_data_parallel
	@echo ""; @echo "=== 02: ZeRO optimizer-state sharding ==="
	./$(BUILD)/02_zero_sharding
	@echo ""; @echo "=== 03: Tensor parallel MLP (Megatron) ==="
	./$(BUILD)/03_tensor_parallel
	@echo ""; @echo "=== 04: 2D device mesh (TP x DP) ==="
	./$(BUILD)/04_2d_mesh

profile-tp:
	mkdir -p profiles
	nsys profile --trace=cuda --output=profiles/tensor_parallel \
	    ./$(BUILD)/03_tensor_parallel

clean:
	rm -rf $(BUILD) profiles

help:
	@echo "Targets:"
	@echo "  all                Build all four programs"
	@echo "  run                Build and run all"
	@echo "  debug              Debug builds (-g -G)"
	@echo "  profile-tp         Nsight Systems on tensor-parallel MLP"
	@echo "  clean              Remove build artefacts"
	@echo ""
	@echo "All programs simulate multi-GPU on a single GPU and verify"
	@echo "correctness against single-GPU references."
	@echo ""
	@echo "Arguments:"
	@echo "  01_data_parallel   <P> <global_batch>   (default 4 256)"
	@echo "  03_tensor_parallel <P>                  (default 4)"
	@echo "  04_2d_mesh         <TP> <DP>            (default 4 4)"
	@echo ""
	@echo "Architecture: make ARCH=sm_80 (A100) / sm_89 (L4/4090) / sm_90 (H100)"
