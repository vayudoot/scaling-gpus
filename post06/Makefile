# Makefile — Scaling GPUs / Post 6: Attention on One GPU

NVCC     := nvcc
ARCH     := native
CFLAGS   := -O3 -arch=$(ARCH) -lineinfo -I include --std=c++17
DBG      := -g -G -O0 -arch=$(ARCH) -I include --std=c++17
BUILD    := build
LIBS     := -lcublas

TARGETS  := 01_softmax 02_attention_naive 03_flash_attention 04_multihead_attention

.PHONY: all run debug clean help

all: $(TARGETS)

$(BUILD):
	mkdir -p $(BUILD)

01_softmax: $(BUILD) src/01_softmax.cu include/utils.cuh
	$(NVCC) $(CFLAGS) src/01_softmax.cu $(LIBS) -o $(BUILD)/01_softmax

02_attention_naive: $(BUILD) src/02_attention_naive.cu include/utils.cuh
	$(NVCC) $(CFLAGS) src/02_attention_naive.cu $(LIBS) -o $(BUILD)/02_attention_naive

03_flash_attention: $(BUILD) src/03_flash_attention.cu include/utils.cuh
	$(NVCC) $(CFLAGS) src/03_flash_attention.cu $(LIBS) -o $(BUILD)/03_flash_attention

04_multihead_attention: $(BUILD) src/04_multihead_attention.cu include/utils.cuh
	$(NVCC) $(CFLAGS) src/04_multihead_attention.cu $(LIBS) -o $(BUILD)/04_multihead_attention

debug: $(BUILD)
	$(NVCC) $(DBG) src/01_softmax.cu             $(LIBS) -o $(BUILD)/01_softmax_dbg
	$(NVCC) $(DBG) src/02_attention_naive.cu     $(LIBS) -o $(BUILD)/02_attention_naive_dbg
	$(NVCC) $(DBG) src/03_flash_attention.cu     $(LIBS) -o $(BUILD)/03_flash_attention_dbg
	$(NVCC) $(DBG) src/04_multihead_attention.cu $(LIBS) -o $(BUILD)/04_multihead_attention_dbg

run: all
	@echo ""; @echo "=== 01: Softmax variants (online, warp-level) ==="
	./$(BUILD)/01_softmax 4096 1024
	@echo ""; @echo "=== 02: Naive attention — N^2 scaling ==="
	./$(BUILD)/02_attention_naive
	@echo ""; @echo "=== 03: FlashAttention — correctness + speedup ==="
	./$(BUILD)/03_flash_attention
	@echo ""; @echo "=== 04: Multi-head attention pipeline ==="
	./$(BUILD)/04_multihead_attention 2 512

# Profiling
profile-naive:
	mkdir -p profiles
	nsys profile --trace=cuda --output=profiles/naive_attn ./$(BUILD)/02_attention_naive

profile-flash:
	mkdir -p profiles
	nsys profile --trace=cuda --output=profiles/flash_attn ./$(BUILD)/03_flash_attention

ncu-softmax:
	ncu --set full --kernel-name softmaxOnline ./$(BUILD)/01_softmax_dbg 1024 512

ncu-flash:
	ncu --set full --kernel-name flashAttentionFwd \
	    --metrics dram__bytes.sum,l1tex__t_bytes.sum \
	    ./$(BUILD)/03_flash_attention

ncu-naive-softmax:
	ncu --metrics dram__bytes.sum \
	    --kernel-name softmaxCausal ./$(BUILD)/02_attention_naive_dbg

clean:
	rm -rf $(BUILD) profiles

help:
	@echo "Targets:"
	@echo "  all                      Build all four programs"
	@echo "  run                      Build and run all benchmarks"
	@echo "  debug                    Debug builds (-g -G)"
	@echo "  profile-naive            Nsight Systems on naive attention"
	@echo "  profile-flash            Nsight Systems on FlashAttention"
	@echo "  ncu-softmax              Nsight Compute on online softmax"
	@echo "  ncu-flash                Nsight Compute on FlashAttention (HBM traffic)"
	@echo "  ncu-naive-softmax        Nsight Compute on naive softmax (shows N^2 reads)"
	@echo "  clean                    Remove build artefacts"
	@echo ""
	@echo "Architecture: make ARCH=sm_80 (A100) / sm_89 (L4/4090) / sm_90 (H100)"
	@echo ""
	@echo "Arguments:"
	@echo "  01_softmax        <rows> <cols>          (default 4096 1024)"
	@echo "  04_multihead_attention <batch> <seq>     (default 2 512)"
