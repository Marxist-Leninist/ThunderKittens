#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
INCLUDE_DIR=${TK_INCLUDE_DIR:-include}
if [[ ! -f "$INCLUDE_DIR/sm75.cuh" && -f ../../../include/sm75.cuh ]]; then
  INCLUDE_DIR=../../../include
fi
# CPU-only compilation; never invokes or benchmarks a GPU.
nvcc -O3 --fmad=false -std=c++17 --cudart shared \
  -gencode arch=compute_75,code=sm_75 -Xcompiler=-fPIC \
  -I"$INCLUDE_DIR" -shared src/rmsnorm_sm75.cu -o build/librmsnorm_sm75.so
