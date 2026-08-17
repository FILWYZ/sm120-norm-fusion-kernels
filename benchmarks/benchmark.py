import argparse
import json
import statistics
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

from sm120_rmsnorm import (
    fused_add_rms_norm,
    fused_add_rms_norm_out,
    fused_add_rms_norm_reference,
    rms_norm,
    rms_norm_out,
    rms_norm_reference,
)


def latency_us(function: Callable[[], object], warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def dtype_from_name(name: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


def run_case(
    tokens: int,
    hidden: int,
    dtype: torch.dtype,
    warmup: int,
    iterations: int,
    include_compile: bool,
) -> list[dict]:
    x = torch.randn((tokens, hidden), device="cuda", dtype=dtype)
    residual = torch.randn_like(x)
    weight = torch.randn((hidden,), device="cuda", dtype=dtype)
    rms_output = torch.empty_like(x)
    fused_output = torch.empty_like(x)
    residual_output = torch.empty_like(x)

    def torch_native_fused() -> tuple[torch.Tensor, torch.Tensor]:
        residual_out = x + residual
        return F.rms_norm(residual_out, (hidden,), weight, 1e-6), residual_out

    implementations: dict[str, Callable[[], object]] = {
        "torch_rms": lambda: rms_norm_reference(x, weight),
        "torch_native_rms": lambda: F.rms_norm(x, (hidden,), weight, 1e-6),
        "cuda_v1_shared": lambda: rms_norm(x, weight, version=1),
        "cuda_v2_warp": lambda: rms_norm(x, weight, version=2),
        "cuda_v2_warp_out": lambda: rms_norm_out(x, weight, rms_output, version=2),
        "torch_fused_reference": lambda: fused_add_rms_norm_reference(x, residual, weight),
        "torch_native_fused": torch_native_fused,
        "cuda_fused_v2": lambda: fused_add_rms_norm(x, residual, weight, version=2),
        "cuda_fused_v2_out": lambda: fused_add_rms_norm_out(
            x, residual, weight, fused_output, residual_output, version=2
        ),
    }
    if hidden == 1024:
        implementations["cuda_fused_v3_h1024"] = lambda: fused_add_rms_norm(
            x, residual, weight, version=3
        )
        implementations["cuda_fused_v3_h1024_out"] = lambda: fused_add_rms_norm_out(
            x, residual, weight, fused_output, residual_output, version=3
        )
        implementations["cuda_fused_v4_packed"] = lambda: fused_add_rms_norm(
            x, residual, weight, version=4
        )
        implementations["cuda_fused_v4_packed_out"] = lambda: fused_add_rms_norm_out(
            x, residual, weight, fused_output, residual_output, version=4
        )
    if include_compile:
        implementations["torch_compile_fused"] = torch.compile(
            torch_native_fused, fullgraph=True
        )

    results = []
    element_size = x.element_size()
    for name, function in implementations.items():
        samples = [latency_us(function, warmup, iterations) for _ in range(3)]
        median_us = statistics.median(samples)
        # Lower-bound traffic estimate. This is intentionally explicit rather
        # than claiming exact physical DRAM traffic from source-level accesses.
        tensor_passes = 3 if "rms" in name and "fused" not in name else 5
        estimated_bytes = tokens * hidden * element_size * tensor_passes
        effective_gbps = estimated_bytes / (median_us * 1e-6) / 1e9
        results.append(
            {
                "implementation": name,
                "tokens": tokens,
                "hidden": hidden,
                "dtype": str(dtype).removeprefix("torch."),
                "latency_us": median_us,
                "effective_bandwidth_gbps_lower_bound": effective_gbps,
                "samples_us": samples,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 4, 16, 64, 256])
    parser.add_argument("--hidden", type=int, nargs="+", default=[768, 1024, 2048, 4096])
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--include-compile", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("benchmark_results/baseline.json"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    results = []
    for tokens in args.tokens:
        for hidden in args.hidden:
            results.extend(
                run_case(
                    tokens,
                    hidden,
                    dtype_from_name(args.dtype),
                    args.warmup,
                    args.iterations,
                    args.include_compile,
                )
            )

    payload = {
        "environment": {
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": torch.cuda.get_device_capability(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
