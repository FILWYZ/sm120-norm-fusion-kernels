# FlashInfer industrial fused baseline

## Scope

This comparison uses the in-place contract shared by FlashInfer and vLLM-style
fused add RMSNorm:

```text
residual = residual + input
input = RMSNorm(residual) * weight
```

Environment: NVIDIA GeForce RTX 5060 Laptop (SM120), PyTorch 2.11.0+cu128,
FlashInfer 0.6.6, CUDA toolkit 12.8, `hidden_size=1024`.

To keep repeated in-place invocations numerically stable without adding reset
kernels to the measured region, the latency benchmark uses zero input and zero
weight after randomized buffer allocation. Both implementations still execute
the same load, reduction, normalization, and store paths. Separate correctness
tests use randomized inputs and weights for FP16, BF16, and FP32.

Protocol:

- five independent Python processes;
- three CUDA Event samples per process;
- 100 warmups and 500 measured iterations per sample;
- table values are medians of the five process medians;
- custom and FlashInfer calls are both preallocated and in-place.

## Results

### FP16

| Tokens | Custom auto | FlashInfer 0.6.6 | Speedup | Latency reduction |
|---:|---:|---:|---:|---:|
| 1 | 5.229 us | 6.027 us | 1.152x | 13.23% |
| 16 | 5.295 us | 5.963 us | 1.126x | 11.20% |
| 64 | 5.831 us | 6.083 us | 1.043x | 4.15% |
| 256 | 6.241 us | 6.428 us | 1.030x | 2.92% |

### BF16

| Tokens | Custom auto | FlashInfer 0.6.6 | Speedup | Latency reduction |
|---:|---:|---:|---:|---:|
| 1 | 5.280 us | 6.085 us | 1.152x | 13.23% |
| 16 | 5.236 us | 6.035 us | 1.153x | 13.24% |
| 64 | 5.815 us | 6.136 us | 1.055x | 5.23% |
| 256 | 6.274 us | 6.396 us | 1.019x | 1.91% |

Raw per-process and aggregate JSON files are stored under
`benchmark_results/flashinfer_repeated/`.

## Dispatch rule

The current SM120 `hidden_size=1024` rule is deliberately simple and keeps all
manual versions available for ablation:

| Rows | Selected kernel |
|---:|---|
| 1-32 | V4 packed |
| 33-192 | V3 register-cached |
| 193+ | V4 packed |
| non-1024 hidden size | V2 generic fallback |

This table is an empirical result for one GPU and one hidden size, not a claim
that the same thresholds are optimal on other architectures.

## SASS evidence

`cuobjdump` with CUDA 13.3 `nvdisasm` shows `LDG.E.128` and `STG.E.128` in the
FP16 and BF16 V4 packed kernel specializations. V3 uses scalar `LDG.E.U16` and
`STG.E.U16` instructions. This confirms that the source-level `alignas(16)`
pack is lowered to 128-bit global-memory instructions.

## Claim boundary

The safe conclusion is that this shape-specialized SM120 implementation is
faster than FlashInfer 0.6.6 for the measured FP16/BF16 shapes. It is not a
general claim against every vLLM, FlashInfer, Inductor, GPU, dtype, or hidden
size. The remaining system-level question is whether replacing RMSNorm in an
LLM engine produces measurable end-to-end decode improvement.
