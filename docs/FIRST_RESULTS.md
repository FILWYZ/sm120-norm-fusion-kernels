# First SM120 Results

## Environment

- GPU: NVIDIA GeForce RTX 5060 Laptop, compute capability 12.0, 8 GiB
- Driver: 610.74
- Build toolkit: CUDA 12.8 (`/usr/local/cuda-12.8`)
- PyTorch: 2.11.0+cu128
- Accumulation: FP32
- Shape: `hidden_size=1024`, tokens `{1, 16, 64, 256}`
- Timing: 100 warmups, 1,000 iterations, three CUDA-event samples; table reports median

The comparison uses `torch.nn.functional.rms_norm` after a separate residual
addition as the PyTorch-native baseline. Preallocated custom `out` variants are
reported separately from allocation-inclusive operator calls.

## FP16 latency

| Tokens | PyTorch native | CUDA V2 generic | CUDA V3 register cached | CUDA V4 packed | Best speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 17.526 μs | 6.232 μs | 5.437 μs | **5.147 μs** | **3.40×** |
| 16 | 17.306 μs | 6.187 μs | 5.378 μs | **5.076 μs** | **3.41×** |
| 64 | 17.286 μs | 6.200 μs | **5.666 μs** | 5.722 μs | **3.05×** |
| 256 | 17.202 μs | 7.573 μs | 6.473 μs | **6.315 μs** | **2.72×** |

## BF16 latency

| Tokens | PyTorch native | CUDA V2 generic | CUDA V3 register cached | CUDA V4 packed | Best speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 17.001 μs | 6.208 μs | 5.357 μs | **5.248 μs** | **3.24×** |
| 16 | 17.202 μs | 6.117 μs | 5.365 μs | **5.143 μs** | **3.34×** |
| 64 | 17.069 μs | 6.290 μs | **5.642 μs** | 5.743 μs | **3.03×** |
| 256 | 16.754 μs | 7.677 μs | 6.658 μs | **6.616 μs** | **2.53×** |

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
At 64 tokens V3 remains slightly faster in repeated operator-level timings, so
the eventual dispatcher should be shape-aware instead of always selecting V4.

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
```

These are first-machine results, not portable performance claims. Before resume
use, rerun the fixed protocol in independent processes and report median plus
dispersion.

