# Makefile -- Scaling GPUs / Post 8: Multi-GPU Infrastructure

NVCC    := nvcc
ARCH    := native
CFLAGS  := -O3 -arch=$(ARCH) -lineinfo -I include --std=c++17
DBG     := -g -G -O0 -arch=$(ARCH) -I include --std=c++17
BUILD   := build

# NCCL auto-detection: link -lnccl only if the header is found in the include path.
# Avoids embedding '#' in the Makefile (which Make treats as a comment).
NCCL_AVAILABLE := $(shell printf 'int main(){return 0;}' > /tmp/_nccltest.cu 2>/dev/null && \
                   $(NVCC) -E -include nccl.h /tmp/_nccltest.cu >/dev/null 2>&1 && \
                   echo yes || echo no; rm -f /tmp/_nccltest.cu)
ifeq ($(NCCL_AVAILABLE),yes)
    NCCL_LIB := -lnccl
else
    NCCL_LIB :=
endif

TARGETS := 01_p2p_bandwidth 02_ring_allreduce 03_nccl_collectives 04_comm_cost_model

.PHONY: all run debug clean help nccl-status

all: nccl-status $(TARGETS)

$(BUILD):
	mkdir -p $(BUILD)

nccl-status:
	@echo "NCCL detected: $(NCCL_AVAILABLE)  (03_nccl_collectives uses it if yes)"

01_p2p_bandwidth: $(BUILD) src/01_p2p_bandwidth.cu include/utils.cuh
	$(NVCC) $(CFLAGS) src/01_p2p_bandwidth.cu -o $(BUILD)/01_p2p_bandwidth

02_ring_allreduce: $(BUILD) src/02_ring_allreduce.cu include/utils.cuh
	$(NVCC) $(CFLAGS) src/02_ring_allreduce.cu -o $(BUILD)/02_ring_allreduce

03_nccl_collectives: $(BUILD) src/03_nccl_collectives.cu include/utils.cuh
	$(NVCC) $(CFLAGS) src/03_nccl_collectives.cu $(NCCL_LIB) -o $(BUILD)/03_nccl_collectives

04_comm_cost_model: $(BUILD) src/04_comm_cost_model.cu include/utils.cuh
	$(NVCC) $(CFLAGS) src/04_comm_cost_model.cu -o $(BUILD)/04_comm_cost_model

debug: $(BUILD)
	$(NVCC) $(DBG) src/01_p2p_bandwidth.cu    -o $(BUILD)/01_p2p_bandwidth_dbg
	$(NVCC) $(DBG) src/02_ring_allreduce.cu   -o $(BUILD)/02_ring_allreduce_dbg
	$(NVCC) $(DBG) src/03_nccl_collectives.cu $(NCCL_LIB) -o $(BUILD)/03_nccl_collectives_dbg
	$(NVCC) $(DBG) src/04_comm_cost_model.cu  -o $(BUILD)/04_comm_cost_model_dbg

run: all
	@echo ""; @echo "=== 01: P2P / NVLink bandwidth ==="
	./$(BUILD)/01_p2p_bandwidth
	@echo ""; @echo "=== 02: Ring-AllReduce from scratch ==="
	./$(BUILD)/02_ring_allreduce
	@echo ""; @echo "=== 03: NCCL collectives ==="
	./$(BUILD)/03_nccl_collectives
	@echo ""; @echo "=== 04: Communication cost model ==="
	./$(BUILD)/04_comm_cost_model

# Profiling
profile-allreduce:
	mkdir -p profiles
	nsys profile --trace=cuda --output=profiles/ring_allreduce \
	    ./$(BUILD)/02_ring_allreduce

profile-nccl:
	mkdir -p profiles
	nsys profile --trace=cuda,nvtx --output=profiles/nccl \
	    ./$(BUILD)/03_nccl_collectives

clean:
	rm -rf $(BUILD) profiles

help:
	@echo "Targets:"
	@echo "  all                  Build all four programs"
	@echo "  run                  Build and run all"
	@echo "  debug                Debug builds (-g -G)"
	@echo "  profile-allreduce    Nsight Systems on ring-AllReduce"
	@echo "  profile-nccl         Nsight Systems on NCCL collectives"
	@echo "  clean                Remove build artefacts"
	@echo ""
	@echo "NCCL: auto-detected. Install with:"
	@echo "  sudo apt install libnccl2 libnccl-dev"
	@echo "  conda install -c conda-forge nccl"
	@echo ""
	@echo "Most programs work on a single GPU (with simulation/fallback)."
	@echo "Programs 01-03 show real multi-GPU behaviour on 2+ GPU nodes."
	@echo ""
	@echo "Architecture: make ARCH=sm_80 (A100) / sm_89 (L4/4090) / sm_90 (H100)"
