# Makefile -- Scaling GPUs / Post 10: Pipeline Parallelism and MoE

NVCC    := nvcc
ARCH    := native
CFLAGS  := -O3 -arch=$(ARCH) -lineinfo -I include --std=c++17
DBG     := -g -G -O0 -arch=$(ARCH) -I include --std=c++17
BUILD   := build
LIBS    := -lcublas

TARGETS := 01_pipeline_schedule 02_pipeline_parallel 03_moe_layer 04_alltoall_cost

.PHONY: all run debug clean help

all: $(TARGETS)

$(BUILD):
	mkdir -p $(BUILD)

01_pipeline_schedule: $(BUILD) src/01_pipeline_schedule.cu include/utils.cuh
	$(NVCC) $(CFLAGS) src/01_pipeline_schedule.cu $(LIBS) -o $(BUILD)/01_pipeline_schedule

02_pipeline_parallel: $(BUILD) src/02_pipeline_parallel.cu include/utils.cuh
	$(NVCC) $(CFLAGS) src/02_pipeline_parallel.cu $(LIBS) -o $(BUILD)/02_pipeline_parallel

03_moe_layer: $(BUILD) src/03_moe_layer.cu include/utils.cuh
	$(NVCC) $(CFLAGS) src/03_moe_layer.cu $(LIBS) -o $(BUILD)/03_moe_layer

04_alltoall_cost: $(BUILD) src/04_alltoall_cost.cu include/utils.cuh
	$(NVCC) $(CFLAGS) src/04_alltoall_cost.cu $(LIBS) -o $(BUILD)/04_alltoall_cost

debug: $(BUILD)
	$(NVCC) $(DBG) src/01_pipeline_schedule.cu $(LIBS) -o $(BUILD)/01_pipeline_schedule_dbg
	$(NVCC) $(DBG) src/02_pipeline_parallel.cu $(LIBS) -o $(BUILD)/02_pipeline_parallel_dbg
	$(NVCC) $(DBG) src/03_moe_layer.cu         $(LIBS) -o $(BUILD)/03_moe_layer_dbg
	$(NVCC) $(DBG) src/04_alltoall_cost.cu     $(LIBS) -o $(BUILD)/04_alltoall_cost_dbg

run: all
	@echo ""
	@echo "=== 01: GPipe vs 1F1B schedule simulator ==="
	./$(BUILD)/01_pipeline_schedule
	@echo ""
	@echo "=== 02: Real pipelined execution (streams) ==="
	./$(BUILD)/02_pipeline_parallel
	@echo ""
	@echo "=== 03: MoE layer (router, experts, capacity) ==="
	./$(BUILD)/03_moe_layer
	@echo ""
	@echo "=== 04: AllToAll simulation + EP cost model ==="
	./$(BUILD)/04_alltoall_cost

profile-pipeline:
	mkdir -p profiles
	nsys profile --trace=cuda --output=profiles/pipeline \
	    ./$(BUILD)/02_pipeline_parallel

profile-moe:
	mkdir -p profiles
	nsys profile --trace=cuda --output=profiles/moe \
	    ./$(BUILD)/03_moe_layer

clean:
	rm -rf $(BUILD) profiles

help:
	@echo "Targets:"
	@echo "  all                 Build all four programs"
	@echo "  run                 Build and run all"
	@echo "  debug               Debug builds (-g -G)"
	@echo "  profile-pipeline    Nsight Systems on the streams pipeline"
	@echo "  profile-moe         Nsight Systems on the MoE layer"
	@echo "  clean               Remove build artefacts"
	@echo ""
	@echo "Arguments:"
	@echo "  01_pipeline_schedule <P> <M>    (default 4 6)"
	@echo "  02_pipeline_parallel <M>        (default 4, max 16)"
	@echo ""
	@echo "All programs run on a single GPU (01 needs no GPU at all)."
	@echo "Architecture: make ARCH=sm_80 (A100) / sm_89 (L4/4090) / sm_90 (H100)"
