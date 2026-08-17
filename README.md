# SM120 Norm Fusion Kernels

CUDA C++ implementations of RMSNorm and decode-time Q/K Norm + RoPE + paged KV
store fusion for LLM inference, tuned for Qwen3-0.6B on an NVIDIA RTX 5060
Laptop GPU (Blackwell SM120).

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
| Auto | measured SM120 shape dispatch | choose V4/V3/V4 by row-count bucket; fall back to V2 |

The fused operator also exposes an in-place API matching FlashInfer/vLLM
semantics: `residual += input`, then `input = RMSNorm(residual) * weight`.

## Qwen3 decode fusion

The production fast path fuses four operations at the real engine boundary:

1. per-head Q/K RMSNorm with FP32 accumulation;
2. half-split RoPE in FP32 after activation-dtype rounding;
3. rotated K scatter into a 16-token paged cache;
4. V scatter into the matching cache slot.

One warp owns one 128-wide head and keeps four values per lane in registers.
The repository also keeps a four-warp/block ablation: it reaches 100%
theoretical occupancy, but did not improve the repeated end-to-end workload.
The integration therefore dispatches the one-warp kernel only for batch-1
decode and retains PyTorch Inductor for prefill and wider decode batches.

Against the equivalent `torch.compile` Q/K Norm+RoPE microbenchmark, the
standalone fusion is 4.9–8.1x faster at 1–256 tokens. In nano-vLLM's Qwen3-0.6B
CUDA Graph path, five isolated and order-balanced A/B processes show batch-1
decode throughput gains of 1.77% and 2.31% for 64- and 256-token prompts;
output throughput improves 1.24% and 1.72%. See
[QK Norm+RoPE+KV results](docs/QK_NORM_ROPE_KV_RESULTS.md).

## Industrial fused baseline

Against FlashInfer 0.6.6 `fused_add_rmsnorm` with identical in-place semantics,
the auto-dispatched custom kernel lowers median operator latency by 2.9–13.2%
for FP16 and 1.9–13.2% for BF16 at `hidden_size=1024`, tokens 1–256. Results are
medians across five independent Python processes; each process uses three CUDA
Event samples, 100 warmups, and 500 measured iterations. See
[the industrial-baseline report](docs/INDUSTRIAL_BASELINE.md).

CUDA 13.3 `nvdisasm` confirms that the CUDA 12.8-built V4 FP16/BF16 kernels
contain `LDG.E.128` and `STG.E.128`. Run `bash scripts/check_sass.sh` to verify
the generated machine instructions locally.

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
bash scripts/check_sass.sh

python -m benchmarks.qk_norm_rope \
  --dtype bf16 \
  --output benchmark_results/qk_norm_rope_bf16.json

python -m benchmarks.profile_qk_norm_rope \
  --tokens 1 --version 1 --store-kv
```

## Correctness contract

- contiguous CUDA tensors;
- matching input/residual/weight dtype;
- FP16, BF16, and FP32 input with FP32 accumulation;
- arbitrary leading dimensions;
- generic V1/V2 supports non-multiple hidden sizes;
- V3/V4 intentionally require `hidden_size=1024`; auto dispatch falls back to
  V2 for other hidden sizes.
- 180 GPU correctness tests cover allocating, preallocated, in-place, manual,
  and auto-dispatched paths.

## Attribution

The project structure and kernels are independently implemented for this
experiment. Useful comparison material includes vLLM's fused RMSNorm operator,
LeetCUDA's RMSNorm exercises, and llm.c's progressive kernel-optimization style.
