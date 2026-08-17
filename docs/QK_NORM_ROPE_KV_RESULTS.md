# Qwen3 Q/K Norm + RoPE + paged KV fusion

## Contract and implementation

The optimized Qwen3-0.6B shape is Q heads `[T, 16, 128]`, K/V heads
`[T, 8, 128]`, BF16 activation/weight, FP32 RoPE cache, int64 positions and
int32 paged-cache slot mapping. The kernel preserves the original numerical
order: FP32 RMS reduction, activation-dtype rounding after RMSNorm, then FP32
half-split RoPE and final activation-dtype rounding.

The decode operator mutates Q/K in place and scatters rotated K plus V into
`[blocks, page_size, 8, 128]` cache storage. A PyTorch custom-op schema declares
all mutated tensors, and a FakeTensor implementation allows
`torch.compile(fullgraph=True)` and CUDA Graph capture.

## Microbenchmark

Environment: RTX 5060 Laptop (SM120), PyTorch 2.11.0+cu128, CUDA toolkit 12.8,
BF16, 100 warmups, 1,000 iterations, three CUDA Event samples.

| Tokens | `torch.compile` | fused Q/K Norm+RoPE | Speedup |
|---:|---:|---:|---:|
| 1 | 46.34 us | 9.36 us | 4.95x |
| 4 | 55.61 us | 7.05 us | 7.89x |
| 16 | 53.32 us | 6.62 us | 8.06x |
| 64 | 54.65 us | 7.11 us | 7.69x |
| 256 | 54.17 us | 11.03 us | 4.91x |

This comparison isolates Q/K Norm+RoPE. The production operator additionally
removes the separate Triton KV append launch and K reread.

## Nsight Compute and SASS

For 64 BF16 tokens, the register-only one-warp Q/K kernel takes 4.48 us versus
4.86 us for the two-warp shared-reduction version. At 256 tokens the result is
10.40 us versus 13.47 us. The selected SASS contains `SHFL.DOWN`, `SHFL.IDX`,
`MUFU.RSQ`, global loads/stores, and no block barrier.

The KV-fused ablation compares one and four warps per block. NCU reports
2.82/2.94 us for one warp and 2.66/2.75 us for four warps at T=1/T=8. Despite
the lower isolated kernel time and 100% theoretical occupancy, the four-warp
version did not produce a stable end-to-end gain. The production dispatch uses
the end-to-end-validated one-warp version; both versions remain reproducible.

## End-to-end dispatch result

Protocol: Qwen3-0.6B BF16, FlashInfer attention, 16-token pages, CUDA Graph
decode, five independent processes per backend, alternating A/B order, two
measurements per workload, and median of process means.

| Requests | Prompt | Decode change | Output throughput change |
|---:|---:|---:|---:|
| 1 | 64 | +1.77% | +1.24% |
| 1 | 256 | +2.31% | +1.72% |

Prefill and decode batches wider than one use the existing Inductor path. Their
reported cross-process deltas are treated as measurement noise, not as kernel
speedups. This narrow dispatch prevents the prefill regressions observed when
the custom operator was enabled globally.

## Verification

- 180 CUDA correctness tests across FP16/BF16/FP32, shapes and both KV versions;
- 41 nano-vLLM tests plus 5 subtests;
- `torch.compile(fullgraph=True)` custom-op smoke test;
- CUDA Graph capture and replay in the real engine;
- Compute Sanitizer memcheck: 0 errors;
- Compute Sanitizer racecheck: 0 hazards;
- CUDA 13.3 NCU and cubin disassembly over CUDA 12.8-built SM120 code.

The claim is deliberately limited to the recorded model, GPU, software stack
and workload. It is not a general performance claim against official vLLM,
FlashInfer, Inductor, or other architectures.
