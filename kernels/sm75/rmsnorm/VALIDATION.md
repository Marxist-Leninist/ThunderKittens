# Validation record

Source base: upstream ThunderKittens commit `67845f5fa48d05343dbbc1ba1403f11061f08d2d`.
Date: 2026-11-11.

## Passed
- Standalone CUDA build: explicit `compute_75/sm_75`, CUDA 12.8, G++11.4.
- Fork-integrated `make -C kernels/sm75/rmsnorm`: exit 0.
- PTX assembler reports SM75 code generation. Sample fused forward: 29 registers,
  36 bytes shared memory, no spills. No TMA/cp.async/WGMMA calls.
- Python syntax compilation of wrapper and all test/benchmark scripts.
- Makefile dry-run for the new SM75 target.
- Independent static review of forward/backward formulas, synchronization and
  wrapper behavior. NaN propagation defect found during review was fixed and
  regression cases added.

## Pending, not passed
- GPU forward/gradient numerical tests.
- Memory-safety sanitizer run.
- GPU benchmarks, full context shapes, and whole-training-step impact.

A bounded 180-second readiness check sampled the existing GTX 1660 SUPER.
Active workloads never provided two consecutive samples at <=15% utilization
and >=1024 MiB free. The final samples had around 661 MiB free and 57–74%
utilization. The launcher correctly skipped all kernel tests. A successful
compile is not runtime correctness or a speedup result.

The training GPU was not used for tests, and no live service, training process,
checkpoint or production source was changed. No new compute was rented.

## Reproduce when hardware is available
```sh
cd kernels/sm75/rmsnorm
make
python3 tests/check_correctness.py --output correctness.json
compute-sanitizer --tool memcheck python3 tests/check_correctness.py
python3 tests/benchmark.py --output benchmark.json --rows 8 --repeats 10
```

Only report speedups after recording device/load conditions and inspecting
correctness output. The benchmark compares the exact unfused PyTorch expression
with this operator at the specified synthetic shape, not a complete model.
