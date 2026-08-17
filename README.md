# SM120 Fused RMSNorm

CUDA C++ implementations of RMSNorm and fused residual-add RMSNorm for LLM
inference, initially tuned for Qwen3-0.6B (`hidden_size=1024`) on an NVIDIA RTX
5060 Laptop GPU (Blackwell SM120).

This repository is a performance-engineering project, not a claim that a custom
kernel always beats PyTorch Inductor or vLLM. Every optimization is kept as a
separate version and evaluated with correctness tests, repeated CUDA-event
measurements, and Nsight Compute.

## Implemented versions

| Version | Kernel | Purpose |
|---|---|---|
| V1 | scalar loads + shared-memory tree reduction | understandable CUDA baseline |
| V2 | hierarchical warp-shuffle reduction | reduce synchronization/shared-memory traffic |
| V2 fused | residual add + generic RMSNorm | remove one launch and expose fusion benefit |
| V3 fused | 1024-wide register-cached fast path | retain four values/thread across reduction |
| V4 fused | dtype-aware 128-bit packed fast path | reduce load/store instruction count |

The next milestone is an empirical SM120 dispatch table across token counts and
hidden sizes, based on repeated measurements rather than a fixed version choice.

## First measured result

On RTX 5060 Laptop/SM120 with PyTorch 2.11.0+cu128, the best preallocated custom
kernel takes 5.08–6.36 μs for FP16 `hidden_size=1024`, tokens 1–256, versus
16.82–17.53 μs for PyTorch-native residual add followed by RMSNorm (2.65–3.40×).
BF16 shows 2.67–3.27×. These ranges use medians from five independent Python
processes. See [the first-results report](docs/FIRST_RESULTS.md) for
the exact protocol, all values, limitations, and Nsight Compute trade-offs.

## Build in WSL

Requirements: NVIDIA driver visible in WSL, CUDA toolkit/NVCC, Python 3.10+,
PyTorch built for CUDA, Ninja, and a compiler compatible with the installed
PyTorch release.

```bash
python -m pip install -U pip ninja pytest
TORCH_CUDA_ARCH_LIST=12.0 python -m pip install -e . --no-build-isolation
pytest -q
```

If the installed PyTorch build cannot compile SM120 device code, first verify:

```bash
nvidia-smi
nvcc --version
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.get_device_name(), torch.cuda.get_device_capability())
PY
```

## Benchmark

```bash
python -m benchmarks.benchmark \
  --tokens 1 4 16 64 256 \
  --hidden 768 1024 2048 4096 \
  --dtype fp16 \
  --output benchmark_results/fp16.json
```

The reported effective bandwidth is a documented source-level lower-bound
estimate. Nsight Compute counters should be used for hardware-level conclusions.

```bash
bash scripts/profile_ncu.sh
```

## Correctness contract

- contiguous CUDA tensors;
- matching input/residual/weight dtype;
- FP16, BF16, and FP32 input with FP32 accumulation;
- arbitrary leading dimensions;
- generic V1/V2 supports non-multiple hidden sizes;
- V3 intentionally requires `hidden_size=1024`.

## Attribution

The project structure and kernels are independently implemented for this
experiment. Useful comparison material includes vLLM's fused RMSNorm operator,
LeetCUDA's RMSNorm exercises, and llm.c's progressive kernel-optimization style.
