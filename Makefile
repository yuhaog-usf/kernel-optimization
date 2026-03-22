# ─── Fused GEMM+bias+ReLU Kernel Optimization ────────────────────
# Target: NVIDIA A100 (sm_80)
#
# Usage:
#   make all        — build all 3 versions
#   make run        — run all 3 and compare
#   make clean      — remove binaries
#   make run M=2048 N=2048 K=2048  — custom matrix size

CUDA_HOME ?= /usr/local/cuda
NVCC = $(CUDA_HOME)/bin/nvcc

# A100 = sm_80
ARCH = -gencode arch=compute_80,code=sm_80

# Flags
NVCCFLAGS = $(ARCH) -O3 -std=c++17 -lineinfo
LDFLAGS = -lcublas

# Matrix size (overridable)
M ?= 1024
N ?= 1024
K ?= 1024

SRC_DIR = src
BIN_DIR = bin

TARGETS = $(BIN_DIR)/v1_separate $(BIN_DIR)/v2_fused_serial $(BIN_DIR)/v3_fused_concurrent

.PHONY: all run clean

all: $(TARGETS)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/v1_separate: $(SRC_DIR)/v1_separate.cu $(SRC_DIR)/common.cuh | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDFLAGS)

$(BIN_DIR)/v2_fused_serial: $(SRC_DIR)/v2_fused_serial.cu $(SRC_DIR)/common.cuh | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(BIN_DIR)/v3_fused_concurrent: $(SRC_DIR)/v3_fused_concurrent.cu $(SRC_DIR)/common.cuh | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

run: all
	@echo "════════════════════════════════════════════════════════"
	@echo " GEMM + bias + ReLU  (M=$(M), N=$(N), K=$(K))"
	@echo "════════════════════════════════════════════════════════"
	@echo ""
	cd $(BIN_DIR) && ./v1_separate $(M) $(N) $(K)
	@echo ""
	cd $(BIN_DIR) && ./v2_fused_serial $(M) $(N) $(K)
	@echo ""
	cd $(BIN_DIR) && ./v3_fused_concurrent $(M) $(N) $(K)
	@echo ""
	@echo "════════════════════════════════════════════════════════"

clean:
	rm -rf $(BIN_DIR) ref_output.bin
