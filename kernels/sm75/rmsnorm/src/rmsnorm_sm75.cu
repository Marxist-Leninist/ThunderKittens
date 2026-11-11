// Native CUDA SM75 RMSNorm primitives for the AGILLM FP32 numerical contract.
// Original implementation, inspired by kernel fusion rather than a TK fork.
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <climits>
#include "sm75.cuh"

namespace {
constexpr int THREADS = 256;
constexpr int ROWS_PER_PART = 32;

__device__ __forceinline__ float block_sum(float v) {
  return kittens::sm75::block_sum<THREADS>(v);
}

template<bool RELU2> __device__ __forceinline__ float activation(float x) {
  if constexpr (RELU2) { float h = x < 0.0f ? 0.0f : x; return h * h; }
  return x;
}

template<bool RELU2>
__global__ void norm_forward(const float* x, const float* gamma, float* y,
                             float* rstd, int64_t d, float eps) {
  const int64_t row = blockIdx.x;
  const int64_t base = row * d;
  float sq = 0.0f;
  for (int64_t j = threadIdx.x; j < d; j += THREADS) {
    float h = activation<RELU2>(x[base+j]);
    sq += h * h;
  }
  sq = block_sum(sq);
  __shared__ float r;
  if (threadIdx.x == 0) { r = rsqrtf(sq / float(d) + eps); rstd[row] = r; }
  __syncthreads();
  for (int64_t j = threadIdx.x; j < d; j += THREADS)
    y[base+j] = (activation<RELU2>(x[base+j]) * r) * gamma[j];
}

template<bool RELU2>
__global__ void norm_dx(const float* x, const float* gamma, const float* dy,
                       const float* rstd, float* dx, int64_t d) {
  const int64_t row = blockIdx.x, base = row * d;
  const float r = rstd[row];
  float dot = 0.0f;
  for (int64_t j = threadIdx.x; j < d; j += THREADS)
    dot += (dy[base+j] * gamma[j]) * activation<RELU2>(x[base+j]);
  dot = block_sum(dot);
  __shared__ float corr;
  if (threadIdx.x == 0) corr = (dot / float(d)) * (r * r);
  __syncthreads();
  for (int64_t j = threadIdx.x; j < d; j += THREADS) {
    float h = activation<RELU2>(x[base+j]);
    float grad = r * ((dy[base+j] * gamma[j]) - h * corr);
    if constexpr (RELU2) grad *= x[base+j] > 0.0f ? 2.0f * x[base+j] : 0.0f;
    dx[base+j] = grad;
  }
}

template<bool RELU2>
__global__ void gamma_partial(const float* x, const float* dy, const float* rstd,
                             float* partial, int64_t rows, int64_t d) {
  const int64_t j = int64_t(blockIdx.x) * THREADS + threadIdx.x;
  if (j >= d) return;
  const int64_t start = int64_t(blockIdx.y) * ROWS_PER_PART;
  const int64_t end = start + ROWS_PER_PART < rows ? start + ROWS_PER_PART : rows;
  float acc = 0.0f;
  for (int64_t row = start; row < end; ++row)
    acc += (dy[row*d+j] * activation<RELU2>(x[row*d+j])) * rstd[row];
  partial[int64_t(blockIdx.y)*d+j] = acc;
}

__global__ void gamma_finish(const float* partial, float* dg, int64_t parts, int64_t d) {
  const int64_t j = int64_t(blockIdx.x) * THREADS + threadIdx.x;
  if (j >= d) return;
  float acc = 0.0f;
  for (int64_t p = 0; p < parts; ++p) acc += partial[p*d+j];
  dg[j] = acc;
}

bool valid(int64_t rows, int64_t d) {
  return rows > 0 && rows <= INT_MAX && d > 0 && d <= INT_MAX &&
         (rows + ROWS_PER_PART - 1) / ROWS_PER_PART <= 65535;
}
}

extern "C" int sm75_rms_forward(const float* x, const float* gamma, float* y,
    float* rstd, int64_t rows, int64_t d, float eps, int relu2, void* stream_ptr) {
  if (!valid(rows,d) || !std::isfinite(eps) || eps <= 0 || (relu2 != 0 && relu2 != 1))
    return int(cudaErrorInvalidValue);
  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  if (relu2) norm_forward<true><<<rows,THREADS,0,stream>>>(x,gamma,y,rstd,d,eps);
  else norm_forward<false><<<rows,THREADS,0,stream>>>(x,gamma,y,rstd,d,eps);
  return int(cudaGetLastError());
}

extern "C" int sm75_rms_backward(const float* x, const float* gamma, const float* dy,
    const float* rstd, float* dx, float* dg, float* partial,
    int64_t rows, int64_t d, int relu2, void* stream_ptr) {
  if (!valid(rows,d) || (relu2 != 0 && relu2 != 1)) return int(cudaErrorInvalidValue);
  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  if (relu2) norm_dx<true><<<rows,THREADS,0,stream>>>(x,gamma,dy,rstd,dx,d);
  else norm_dx<false><<<rows,THREADS,0,stream>>>(x,gamma,dy,rstd,dx,d);
  auto status = cudaGetLastError();
  if (status != cudaSuccess) return int(status);
  const int64_t parts = (rows + ROWS_PER_PART - 1) / ROWS_PER_PART;
  const dim3 grid((d + THREADS - 1) / THREADS, parts);
  float* dst = parts == 1 ? dg : partial;
  if (!dst) return int(cudaErrorInvalidValue);
  if (relu2) gamma_partial<true><<<grid,THREADS,0,stream>>>(x,dy,rstd,dst,rows,d);
  else gamma_partial<false><<<grid,THREADS,0,stream>>>(x,dy,rstd,dst,rows,d);
  status = cudaGetLastError();
  if (status != cudaSuccess) return int(status);
  if (parts > 1) gamma_finish<<<(d+THREADS-1)/THREADS,THREADS,0,stream>>>(partial,dg,parts,d);
  return int(cudaGetLastError());
}

extern "C" const char* sm75_error_string(int error) {
  return cudaGetErrorString(static_cast<cudaError_t>(error));
}
