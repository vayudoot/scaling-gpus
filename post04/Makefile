# Makefile — Scaling GPUs / Post 4: CUDA Streams and Async Execution

NVCC     := nvcc
ARCH     := native
CFLAGS   := -O3 -arch=$(ARCH) -lineinfo -I include --std=c++17
DBG      := -g -G -O0 -arch=$(ARCH) -I include --std=c++17
BUILD    := build

# NVTX3 is header-only — no -l flag needed for any CUDA version

TARGETS := 01_default_stream 02_async_copy 03_double_buffer 04_events 05_cuda_graphs

.PHONY: all run debug clean profile-all help

all: $(TARGETS)

$(BUILD):
	mkdir -p $(BUILD)

01_default_stream: $(BUILD) src/01_default_stream.cu include/timer.cuh
	$(NVCC) $(CFLAGS) src/01_default_stream.cu -o $(BUILD)/01_default_stream

02_async_copy: $(BUILD) src/02_async_copy.cu include/timer.cuh
	$(NVCC) $(CFLAGS) src/02_async_copy.cu -o $(BUILD)/02_async_copy

03_double_buffer: $(BUILD) src/03_double_buffer.cu include/timer.cuh
	$(NVCC) $(CFLAGS) src/03_double_buffer.cu -o $(BUILD)/03_double_buffer

04_events: $(BUILD) src/04_events.cu include/timer.cuh
	$(NVCC) $(CFLAGS) src/04_events.cu -o $(BUILD)/04_events

05_cuda_graphs: $(BUILD) src/05_cuda_graphs.cu include/timer.cuh
	$(NVCC) $(CFLAGS) src/05_cuda_graphs.cu -o $(BUILD)/05_cuda_graphs

debug: $(BUILD)
	$(NVCC) $(DBG) src/01_default_stream.cu -o $(BUILD)/01_default_stream_dbg
	$(NVCC) $(DBG) src/02_async_copy.cu     -o $(BUILD)/02_async_copy_dbg
	$(NVCC) $(DBG) src/03_double_buffer.cu  -o $(BUILD)/03_double_buffer_dbg
	$(NVCC) $(DBG) src/04_events.cu         -o $(BUILD)/04_events_dbg
	$(NVCC) $(DBG) src/05_cuda_graphs.cu    -o $(BUILD)/05_cuda_graphs_dbg

# ── Run all ───────────────────────────────────────────────────────────────────
run: all
	@echo ""; @echo "═══ 01: Default stream serialisation ═══"
	./$(BUILD)/01_default_stream
	@echo ""; @echo "═══ 02: Async copy — pageable vs pinned ═══"
	./$(BUILD)/02_async_copy
	@echo ""; @echo "═══ 03: Double-buffer pipeline ═══"
	./$(BUILD)/03_double_buffer 20
	@echo ""; @echo "═══ 04: CUDA events — timing and sync ═══"
	./$(BUILD)/04_events
	@echo ""; @echo "═══ 05: CUDA Graphs ═══"
	./$(BUILD)/05_cuda_graphs

# ── Nsight Systems profiles ───────────────────────────────────────────────────
profile-streams:
	mkdir -p profiles
	nsys profile --trace=cuda,nvtx --output=profiles/streams \
	    ./$(BUILD)/01_default_stream

profile-async:
	mkdir -p profiles
	nsys profile --trace=cuda --output=profiles/async_copy \
	    ./$(BUILD)/02_async_copy

profile-pipeline:
	mkdir -p profiles
	nsys profile --trace=cuda,nvtx --output=profiles/pipeline \
	    ./$(BUILD)/03_double_buffer 30

profile-graphs:
	mkdir -p profiles
	nsys profile --trace=cuda --output=profiles/graphs \
	    ./$(BUILD)/05_cuda_graphs

# Key metric: is the GPU idle between launches?
ncu-launch-overhead:
	ncu --metrics \
	    launch__occupancy_limit_warps,\
	    sm__cycles_elapsed.avg.per_second \
	    ./$(BUILD)/05_cuda_graphs

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	rm -rf $(BUILD) profiles

help:
	@echo "Targets:"
	@echo "  all                 Build all programs"
	@echo "  run                 Build and run all benchmarks"
	@echo "  debug               Debug builds (-g -G)"
	@echo "  profile-streams     Nsight Systems — stream serialisation demo"
	@echo "  profile-async       Nsight Systems — async copy overlap"
	@echo "  profile-pipeline    Nsight Systems — double-buffer pipeline (NVTX)"
	@echo "  profile-graphs      Nsight Systems — CUDA Graphs vs eager"
	@echo "  ncu-launch-overhead Nsight Compute — kernel launch cost"
	@echo "  clean               Remove build artefacts"
	@echo ""
	@echo "Architecture:"
	@echo "  make ARCH=sm_80 all   (A100)"
	@echo "  make ARCH=sm_89 all   (RTX 4090 / L4)"
	@echo "  make ARCH=sm_90 all   (H100)"
	@echo ""
	@echo "NVTX note: if 03_double_buffer fails to link, try:"
	@echo "  make CFLAGS='-O3 -arch=\$(ARCH) -I include --std=c++17'"
