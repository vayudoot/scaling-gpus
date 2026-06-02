# Makefile — Scaling GPUs / Post 5: Building a Neural Network on One GPU

NVCC     := nvcc
ARCH     := native
CFLAGS   := -O3 -arch=$(ARCH) -lineinfo -I include --std=c++17
DBG      := -g -G -O0 -arch=$(ARCH) -I include --std=c++17
BUILD    := build

# cuBLAS is required for all programs in this post
LIBS     := -lcublas

TARGETS  := 01_activations 02_mlp_forward 03_backprop 04_training_loop

.PHONY: all run debug clean help

all: $(TARGETS)

$(BUILD):
	mkdir -p $(BUILD)

01_activations: $(BUILD) src/01_activations.cu include/utils.cuh
	$(NVCC) $(CFLAGS) src/01_activations.cu $(LIBS) -o $(BUILD)/01_activations

02_mlp_forward: $(BUILD) src/02_mlp_forward.cu include/utils.cuh
	$(NVCC) $(CFLAGS) src/02_mlp_forward.cu $(LIBS) -o $(BUILD)/02_mlp_forward

03_backprop: $(BUILD) src/03_backprop.cu include/utils.cuh
	$(NVCC) $(CFLAGS) src/03_backprop.cu $(LIBS) -o $(BUILD)/03_backprop

04_training_loop: $(BUILD) src/04_training_loop.cu include/utils.cuh
	$(NVCC) $(CFLAGS) src/04_training_loop.cu $(LIBS) -o $(BUILD)/04_training_loop

debug: $(BUILD)
	$(NVCC) $(DBG) src/01_activations.cu    $(LIBS) -o $(BUILD)/01_activations_dbg
	$(NVCC) $(DBG) src/02_mlp_forward.cu   $(LIBS) -o $(BUILD)/02_mlp_forward_dbg
	$(NVCC) $(DBG) src/03_backprop.cu      $(LIBS) -o $(BUILD)/03_backprop_dbg
	$(NVCC) $(DBG) src/04_training_loop.cu $(LIBS) -o $(BUILD)/04_training_loop_dbg

run: all
	@echo ""; @echo "=== 01: Activation functions ==="
	./$(BUILD)/01_activations
	@echo ""; @echo "=== 02: MLP forward pass ==="
	./$(BUILD)/02_mlp_forward 512
	@echo ""; @echo "=== 03: Backpropagation ==="
	./$(BUILD)/03_backprop 256
	@echo ""; @echo "=== 04: Training loop (500 steps) ==="
	./$(BUILD)/04_training_loop 256 500

# Profile targets
profile-fwd:
	mkdir -p profiles
	nsys profile --trace=cuda --output=profiles/mlp_forward \
	    ./$(BUILD)/02_mlp_forward 512

profile-train:
	mkdir -p profiles
	nsys profile --trace=cuda --output=profiles/training \
	    ./$(BUILD)/04_training_loop 256 50

ncu-activations:
	ncu --set full --kernel-name geluKernel ./$(BUILD)/01_activations

ncu-adam:
	ncu --metrics dram__bytes.sum \
	    --kernel-name adamUpdateKernel ./$(BUILD)/04_training_loop 256 5

clean:
	rm -rf $(BUILD) profiles

help:
	@echo "Targets:"
	@echo "  all                Build all four programs"
	@echo "  run                Build and run all benchmarks"
	@echo "  debug              Debug builds (-g -G -O0)"
	@echo "  profile-fwd        Nsight Systems on MLP forward pass"
	@echo "  profile-train      Nsight Systems on training loop"
	@echo "  ncu-activations    Nsight Compute on GELU kernel"
	@echo "  ncu-adam           Nsight Compute on Adam optimizer kernel"
	@echo "  clean              Remove build artefacts"
	@echo ""
	@echo "Architecture: make ARCH=sm_80 (A100) / sm_89 (L4/4090) / sm_90 (H100)"
	@echo ""
	@echo "Program arguments:"
	@echo "  02_mlp_forward  <batch_size>              (default 512)"
	@echo "  03_backprop     <batch_size>              (default 256)"
	@echo "  04_training_loop <batch_size> <steps>     (default 256 500)"
