# Experimental SM75 FP32 normalization

This fork adds a **scoped Turing backend**, not general SM75 support for every
ThunderKittens tile API or a port of the Megakernels runtime. Existing upstream
Hopper/Blackwell kernels remain unchanged. The new `include/sm75.cuh` provides
portable warp/block FP32 reductions, and this directory uses them for two
training-relevant operations:

- RMSNorm: `y = x * rsqrt(mean(x²) + eps) * gamma`.
- Fused ReLU² then RMSNorm: replace `x` in the same formula with `relu(x)²`.

Both support FP32 forward and first-order gradients for x and gamma. Default
epsilon is 1e-6. No mean-centering, BF16, TMA, cp.async, WGMMA, or tensor cores
are required. The implementation is original native CUDA under the repository's
MIT license, inspired by fusion and small hardware-compatible primitives.

## Build and run

Requirements: NVIDIA Turing SM75, CUDA toolkit (tested compiler 12.8), a C++17+
host compiler, and a CUDA-enabled PyTorch for Python tests. No pybind11 or
Torch C++ extension build is needed; the wrapper uses a small ctypes C ABI.

From this directory in the fork:

```sh
make
python3 tests/check_cpu.py              # CPU contract checks, no GPU validation
python3 tests/check_correctness.py --output correctness.json
python3 tests/benchmark.py --output benchmark.json --rows 8 --repeats 10
```

`bash build.sh` is also supported. Builds compile explicitly for SM75 and do
not launch GPU work. Run tests only on available hardware with adequate memory.
`tests/wait_for_window.py` can make a bounded check for two low-utilization
samples and at least 1 GiB free before the tiny correctness suite. This is an
advisory readiness check, not a reservation or guarantee against other work.

Example:

```python
from sm75_norm import rmsnorm
y = rmsnorm(x, gamma)                 # x: [...,D], gamma: [D], FP32 CUDA
z = rmsnorm(x, gamma, relu2=True)      # fused expert-FFN activation + norm
y.sum().backward()
```

## Contract and limits

- Contiguous FP32 x/gamma, same CUDA device, normalization over the last axis.
- Arbitrary positive widths, including nonmultiples of the warp/block size;
  leading dimensions flatten to rows. Empty leading dimensions are supported.
- Two-stage gamma-gradient reduction uses no floating-point atomics.
- NaNs propagate, including through the fused activation; the kernel must not
  conceal an upstream numerical failure.
- Algebraic equivalence is tested within explicit FP32 tolerances, not bitwise
  equality. Build deliberately avoids fast math and fused multiply-add.
- First-order gradients only. Double backward, vmap, autocast policy, exported
  graphs and torch.compile custom-op registration are not implemented.
- Explicit `fallback=True` runs the PyTorch reference for unsupported
  dtype/layout/CPU inputs. Its statistics deliberately use float32, matching
  the target expression even for float64 input; it is not a high-precision API.
- `forward_out` is a low-level inference/detached-buffer API with no autograd
  mutation tracking. Never use it to overwrite tensors in a live autograd graph.
- Operations launch on the current stream for the tensor's device. For
  cross-stream handoffs, callers must establish producer/consumer dependencies
  and protect tensor lifetime with the usual wait/record_stream conventions.
- These kernels are not installed into any trainer. Checkpoints, optimizer
  state, data and running services are untouched.

## Validation status

The standalone source has compiled successfully for SM75 with CUDA 12.8 and
G++11. Hardware numerical results and benchmark status are recorded separately
in `VALIDATION.md`; do not infer a performance improvement from compilation.

The bounded suite covers both operations, actual widths 8192/32768, irregular
widths, forward/dx/dgamma against FP32 autograd and independent CPU float64,
multi-part reductions, zero/tiny/large/constant values, NaNs, negative-only
fused rows, nondefault epsilon, nondefault streams, rank1/rank3 inputs,
noncontiguous upstream gradients, input guards and empty rows.

Benchmark "forward" includes autograd bookkeeping because inputs require
gradients. "forward_and_backward" includes both passes. Timings are operator
microbenchmarks, not context2048 or whole-training-step results; overlapping
GPU workloads invalidate a clean speedup claim. Full context2048×32768 arrays
are 256 MiB each in FP32, so this suite intentionally uses smaller row counts.
