import argparse
import json
import statistics
from pathlib import Path
from typing import Callable

import torch

from sm120_rmsnorm import fused_qk_rms_norm_rope, qk_rms_norm_rope_reference


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


def make_cache(max_position: int, device: torch.device) -> torch.Tensor:
    inv_freq = 1.0 / (
        1_000_000.0
        ** (torch.arange(0, 128, 2, dtype=torch.float32, device=device) / 128)
    )
    positions = torch.arange(max_position, dtype=torch.float32, device=device)
    frequencies = torch.outer(positions, inv_freq)
    return torch.cat((frequencies.cos(), frequencies.sin()), dim=-1).unsqueeze(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 4, 16, 64, 256])
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    device = torch.device("cuda")
    results = []

    for tokens in args.tokens:
        torch.manual_seed(31)
        query = torch.randn((tokens, 16, 128), device=device, dtype=dtype)
        key = torch.randn((tokens, 8, 128), device=device, dtype=dtype)
        query_weight = torch.randn(128, device=device, dtype=dtype)
        key_weight = torch.randn(128, device=device, dtype=dtype)
        positions = torch.arange(tokens, device=device, dtype=torch.int64)
        cache = make_cache(max(tokens, 1), device)

        def eager() -> tuple[torch.Tensor, torch.Tensor]:
            return qk_rms_norm_rope_reference(
                query, key, query_weight, key_weight, positions, cache
            )

        compiled = torch.compile(eager, fullgraph=True)
        custom_query = torch.zeros_like(query)
        custom_key = torch.zeros_like(key)
        custom_v1_query = torch.zeros_like(query)
        custom_v1_key = torch.zeros_like(key)
        zero_query_weight = torch.zeros_like(query_weight)
        zero_key_weight = torch.zeros_like(key_weight)

        def custom() -> None:
            fused_qk_rms_norm_rope(
                custom_query,
                custom_key,
                zero_query_weight,
                zero_key_weight,
                positions,
                cache,
                version=2,
            )

        def custom_v1() -> None:
            fused_qk_rms_norm_rope(
                custom_v1_query,
                custom_v1_key,
                zero_query_weight,
                zero_key_weight,
                positions,
                cache,
                version=1,
            )

        implementations = {
            "torch_eager": eager,
            "torch_compile": compiled,
            "sm120_fused_v1_block64": custom_v1,
            "sm120_fused_v2_warp32": custom,
        }
        for name, function in implementations.items():
            samples = [latency_us(function, args.warmup, args.iterations) for _ in range(3)]
            results.append(
                {
                    "implementation": name,
                    "tokens": tokens,
                    "dtype": args.dtype,
                    "latency_us": statistics.median(samples),
                    "samples_us": samples,
                }
            )

    payload = {
        "environment": {
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": torch.cuda.get_device_capability(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "shape": {"query_heads": 16, "key_heads": 8, "head_dim": 128},
        "protocol": {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "samples": 3,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
