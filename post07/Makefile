# Makefile -- Scaling GPUs / Post 7: Mixed Precision and Quantization

NVCC    := nvcc
ARCH    := native
CFLAGS  := -O3 -arch=$(ARCH) -lineinfo -I include --std=c++17
DBG     := -g -G -O0 -arch=$(ARCH) -I include --std=c++17
BUILD   := build
LIBS    := -lcublas

TARGETS := 01_precision_formats 02_tensor_cores 03_mixed_precision \
           04_quantization 05_w4a16

.PHONY: all run debug clean help

all: $(TARGETS)

$(BUILD):
	mkdir -p $(BUILD)

01_precision_formats: $(BUILD) src/01_precision_formats.cu include/utils.cuh
	$(NVCC) $(CFLAGS) src/01_precision_formats.cu $(LIBS) -o $(BUILD)/01_precision_formats

02_tensor_cores: $(BUILD) src/02_tensor_cores.cu include/utils.cuh
	$(NVCC) $(CFLAGS) src/02_tensor_cores.cu $(LIBS) -o $(BUILD)/02_tensor_cores

03_mixed_precision: $(BUILD) src/03_mixed_precision.cu include/utils.cuh
	$(NVCC) $(CFLAGS) src/03_mixed_precision.cu $(LIBS) -o $(BUILD)/03_mixed_precision

04_quantization: $(BUILD) src/04_quantization.cu include/utils.cuh
	$(NVCC) $(CFLAGS) src/04_quantization.cu $(LIBS) -o $(BUILD)/04_quantization

05_w4a16: $(BUILD) src/05_w4a16.cu include/utils.cuh
	$(NVCC) $(CFLAGS) src/05_w4a16.cu $(LIBS) -o $(BUILD)/05_w4a16

debug: $(BUILD)
	$(NVCC) $(DBG) src/01_precision_formats.cu $(LIBS) -o $(BUILD)/01_precision_formats_dbg
	$(NVCC) $(DBG) src/02_tensor_cores.cu      $(LIBS) -o $(BUILD)/02_tensor_cores_dbg
	$(NVCC) $(DBG) src/03_mixed_precision.cu   $(LIBS) -o $(BUILD)/03_mixed_precision_dbg
	$(NVCC) $(DBG) src/04_quantization.cu      $(LIBS) -o $(BUILD)/04_quantization_dbg
	$(NVCC) $(DBG) src/05_w4a16.cu             $(LIBS) -o $(BUILD)/05_w4a16_dbg

run: all
	@echo ""; @echo "=== 01: Precision formats: overflow, throughput, matmul ==="
	./$(BUILD)/01_precision_formats
	@echo ""; @echo "=== 02: Tensor Cores: WMMA API, correctness, benchmark ==="
	./$(BUILD)/02_tensor_cores
	@echo ""; @echo "=== 03: Mixed precision: FP32 master + FP16 forward ==="
	./$(BUILD)/03_mixed_precision
	@echo ""; @echo "=== 04: INT8 quantization and dp4a matmul ==="
	./$(BUILD)/04_quantization
	@echo ""; @echo "=== 05: W4A16 -- 4-bit weights for LLM decode ==="
	./$(BUILD)/05_w4a16

# Profiling
profile-matmul:
	mkdir -p profiles
	nsys profile --trace=cuda --output=profiles/precision_matmul \
	    ./$(BUILD)/01_precision_formats

ncu-tensor-core-util:
	ncu --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active \
	    ./$(BUILD)/02_tensor_cores

ncu-wmma:
	ncu --set full --kernel-name wmmaMatmul \
	    ./$(BUILD)/02_tensor_cores_dbg

ncu-int8-bw:
	ncu --metrics dram__bytes.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed \
	    ./$(BUILD)/04_quantization_dbg

ncu-w4a16:
	ncu --metrics dram__bytes.sum \
	    --kernel-name w4a16MatmulKernel \
	    ./$(BUILD)/05_w4a16_dbg

clean:
	rm -rf $(BUILD) profiles

help:
	@echo "Targets:"
	@echo "  all                    Build all five programs"
	@echo "  run                    Build and run all benchmarks"
	@echo "  debug                  Debug builds (-g -G)"
	@echo "  ncu-tensor-core-util   Check Tensor Core utilization %"
	@echo "  ncu-wmma               Full Nsight Compute on WMMA kernel"
	@echo "  ncu-int8-bw            INT8 kernel memory bandwidth"
	@echo "  ncu-w4a16              W4A16 weight bytes loaded from HBM"
	@echo "  profile-matmul         Nsight Systems on precision benchmark"
	@echo "  clean                  Remove build artefacts"
	@echo ""
	@echo "Architecture: make ARCH=sm_80 (A100) / sm_89 (L4/4090) / sm_90 (H100)"
	@echo ""
	@echo "Notes:"
	@echo "  - BF16 kernels in 01_precision_formats require sm_80+ (Ampere+)"
	@echo "  - WMMA in 02_tensor_cores requires sm_70+ (Volta+)"
	@echo "  - dp4a in 04_quantization requires sm_61+ (Pascal+)"
