# Scoped Turing lane. Legacy Hopper/Blackwell examples are not SM75-compatible.
ARCH := SM75
SRC := src/rmsnorm_sm75.cu
OUT := build/librmsnorm_sm75.so
CMD := python3 tests/check_correctness.py
CONFIG := standalone
NVCCFLAGS := -shared -Xcompiler=-fPIC --fmad=false --cudart=shared
include ../../common.mk
# Numerical equivalence comes before fast-math optimization in this backend.
NVCCFLAGS := $(filter-out --use_fast_math -lcudart_static,$(NVCCFLAGS))
$(OUT): | build
build:
	mkdir -p build
.PHONY: check benchmark
check: $(OUT)
	python3 tests/check_correctness.py
benchmark: $(OUT)
	python3 tests/benchmark.py
