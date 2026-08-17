# First SM120 Results

## Environment

- GPU: NVIDIA GeForce RTX 5060 Laptop, compute capability 12.0, 8 GiB
- Driver: 610.74
- Build toolkit: CUDA 12.8 (`/usr/local/cuda-12.8`)
- PyTorch: 2.11.0+cu128
- Accumulation: FP32
- Shape: `hidden_size=1024`, tokens `{1, 16, 64, 256}`
- Timing: five independent Python processes; each process uses 100 warmups, 500
  iterations, and three CUDA-event samples. Tables report the median of the five
  per-process medians.

The comparison uses `torch.nn.functional.rms_norm` after a separate residual
addition as the PyTorch-native baseline. Preallocated custom `out` variants are
reported separately from allocation-inclusive operator calls.

## FP16 latency

| Tokens | PyTorch native | CUDA V2 generic | CUDA V3 register cached | CUDA V4 packed | Best speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 17.525 μs | 6.724 μs | 5.628 μs | **5.425 μs** | **3.23×** |
| 16 | 17.256 μs | 6.284 μs | 5.518 μs | **5.077 μs** | **3.40×** |
| 64 | 17.112 μs | 6.505 μs | 6.132 μs | **5.900 μs** | **2.90×** |
| 256 | 16.821 μs | 7.528 μs | **6.357 μs** | 6.377 μs | **2.65×** |

## BF16 latency

| Tokens | PyTorch native | CUDA V2 generic | CUDA V3 register cached | CUDA V4 packed | Best speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 17.265 μs | 6.335 μs | 5.402 μs | **5.277 μs** | **3.27×** |
| 16 | 17.234 μs | 6.178 μs | 5.439 μs | **5.315 μs** | **3.24×** |
| 64 | 17.183 μs | 6.443 μs | 5.905 μs | **5.821 μs** | **2.95×** |
| 256 | 16.965 μs | 7.611 μs | 6.445 μs | **6.363 μs** | **2.67×** |

## Nsight Compute: FP16, tokens=256

| Metric | V3 register cached | V4 128-bit packed |
|---|---:|---:|
| Kernel duration | 5.73 μs | **5.12 μs** |
| Memory throughput | 184.31 GB/s | **206.25 GB/s** |
| DRAM throughput | 42.12% | **47.39%** |
| Registers/thread | **20** | 28 |
| Achieved occupancy | **74.65%** | 51.72% |

V4 reduces pure-kernel duration by about 10.6% and raises measured memory
throughput by about 11.9%. Its 128-bit packing increases register use and lowers
occupancy, but the reduced load/store instruction pressure wins at 256 tokens.
V4 wins most shapes, while FP16 at 256 tokens slightly favors V3 in the
cross-process median. The eventual dispatcher should therefore be shape-aware
instead of always selecting V4.

One FP16 process experienced a system-wide latency spike that affected both
PyTorch and custom implementations. We retain it in the raw per-run data and use
the cross-process median rather than selecting the fastest process.

## Correctness

`59 passed` on GPU across FP16, BF16, and FP32; V1/V2 generic paths include
non-multiple hidden sizes, while V3/V4 enforce `hidden_size=1024`.

## Reproduction

```bash
CUDA_HOME=/usr/local/cuda-12.8 \
PATH=/usr/local/cuda-12.8/bin:$PATH \
TORCH_CUDA_ARCH_LIST=12.0 \
uv pip install --python /path/to/python -e . --no-build-isolation

/path/to/python -m pytest -q
/path/to/python -m benchmarks.benchmark \
  --tokens 1 16 64 256 --hidden 1024 --dtype fp16 \
  --warmup 100 --iterations 1000 \
  --output benchmark_results/fp16_h1024_v4.json

/path/to/python -m benchmarks.run_repeated \
  --dtype fp16 --repeats 5 --warmup 100 --iterations 500
```

These are first-machine results, not portable performance claims. Resume claims
should retain the exact GPU, dtype, shape range, baseline, and repeated-process
protocol.
