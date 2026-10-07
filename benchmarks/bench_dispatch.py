"""Compare SM120 fused-add RMSNorm specializations with alternating order."""
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

import torch
from sm120_rmsnorm import fused_add_rms_norm_inplace


def latency_us(function, iterations):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def environment():
    return {
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }


def run_worker(dtype_name, tokens, rounds, iterations):
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
    results = []
    for count in tokens:
        x = torch.zeros((count, 1024), device="cuda", dtype=dtype)
        residual = torch.zeros_like(x)
        weight = torch.ones((1024,), device="cuda", dtype=dtype)
        functions = {
            "v3": lambda: fused_add_rms_norm_inplace(
                x, residual, weight, version=3
            ),
            "v4": lambda: fused_add_rms_norm_inplace(
                x, residual, weight, version=4
            ),
        }
        for function in functions.values():
            for _ in range(20):
                function()
        torch.cuda.synchronize()
        samples = {name: [] for name in functions}
        for index in range(rounds):
            order = ("v3", "v4") if index % 2 == count % 2 else ("v4", "v3")
            for name in order:
                samples[name].append(latency_us(functions[name], iterations))
        medians = {name: statistics.median(values) for name, values in samples.items()}
        results.append({"tokens": count, "samples_us": samples, "median_us": medians})
    return {"environment": environment(), "dtype": dtype_name, "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], nargs="+", default=["fp16", "bf16"])
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 16, 32, 33, 64, 128, 192, 193, 256])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results/dispatch_paired"))
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--worker-dtype")
    args = parser.parse_args()

    if args.worker_output:
        payload = run_worker(args.worker_dtype, args.tokens, args.rounds, args.iterations)
        args.worker_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = {dtype: [] for dtype in args.dtype}
    for dtype in args.dtype:
        for index in range(args.repeats):
            output = args.output_dir / f"{dtype}_run_{index + 1:02d}.json"
            command = [
                sys.executable, "-m", "benchmarks.bench_dispatch",
                "--worker-output", str(output), "--worker-dtype", dtype,
                "--tokens", *map(str, args.tokens),
                "--rounds", str(args.rounds), "--iterations", str(args.iterations),
            ]
            print(f"{dtype} process {index + 1}/{args.repeats}", flush=True)
            subprocess.run(command, check=True)
            runs[dtype].append(json.loads(output.read_text(encoding="utf-8")))

    summary = {"schema_version": 1, "method": "alternating_paired_v3_v4", "repeats": args.repeats,
               "rounds_per_process": args.rounds, "iterations_per_sample": args.iterations,
               "tokens": args.tokens, "workloads": {}}
    for dtype, payloads in runs.items():
        by_token = {}
        for count in args.tokens:
            v3 = [next(row for row in payload["results"] if row["tokens"] == count)["median_us"]["v3"] for payload in payloads]
            v4 = [next(row for row in payload["results"] if row["tokens"] == count)["median_us"]["v4"] for payload in payloads]
            ratios = [a / b for a, b in zip(v3, v4)]
            by_token[str(count)] = {
                "v3_median_us": statistics.median(v3),
                "v4_median_us": statistics.median(v4),
                "v4_speedup_vs_v3_median": statistics.median(ratios),
                "v3_process_medians_us": v3,
                "v4_process_medians_us": v4,
                "paired_speedup_per_process": ratios,
                "winner": "v4" if statistics.median(ratios) > 1 else "v3",
            }
        summary["workloads"][dtype] = by_token
    summary["environment"] = runs[args.dtype[0]][0]["environment"]
    path = args.output_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"summary={path}")


if __name__ == "__main__":
    main()
