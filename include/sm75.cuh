/**
 * Experimental Turing-safe primitives for the fork's portable FP32 kernels.
 * This is a scoped backend, not a claim that Hopper/Blackwell APIs run on SM75.
 */
#pragma once
#include <cuda_runtime.h>

namespace kittens { namespace sm75 {
constexpr int WARP_THREADS = 32;
constexpr int DEFAULT_BLOCK_THREADS = 256;
constexpr int PORTABLE_SHARED_MEMORY_LIMIT = 48 * 1024;

// Call uniformly from every lane of an active, full warp.
__device__ __forceinline__ float warp_sum(float v) {
    for (int offset = 16; offset; offset >>= 1)
        v += __shfl_down_sync(0xffffffffu,v,offset);
    return v;
}

// Call uniformly from exactly Threads threads. Only thread 0 receives the sum.
// Callers must synchronize before a subsequent invocation reuses shared memory.
template<int Threads>
__device__ __forceinline__ float block_sum(float v) {
    static_assert(Threads > 0 && Threads <= 1024 && Threads % WARP_THREADS == 0);
    __shared__ float sums[Threads / WARP_THREADS];
    v = warp_sum(v);
    if ((threadIdx.x & 31) == 0) sums[threadIdx.x >> 5] = v;
    __syncthreads();
    v = threadIdx.x < Threads / WARP_THREADS ? sums[threadIdx.x] : 0.0f;
    if (threadIdx.x < WARP_THREADS) v = warp_sum(v);
    return v;
}
}}
