"""Summarize repeated RMSNorm shape-matrix JSON files into JSON and Markdown."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    files = sorted(args.input_dir.glob("*_run_*.json"))
    if not files:
        raise SystemExit(f"no *_run_*.json files found in {args.input_dir}")

    grouped: dict[tuple[str, int, int, str], list[float]] = defaultdict(list)
    environments = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        environments.append(payload["environment"])
        for row in payload["results"]:
            key = (row["dtype"], row["tokens"], row["hidden"], row["implementation"])
            grouped[key].append(float(row["latency_us"]))

    runs_by_dtype = defaultdict(set)
    for path in files:
        runs_by_dtype[path.stem.split("_run_")[0]].add(path.stem)
    if len({len(x) for x in runs_by_dtype.values()}) != 1:
        raise SystemExit("each dtype must have the same number of independent run files")

    summary = {}
    md = [
        "# RTX 5060 RMSNorm Shape Performance Map",
        "",
        "Generated from the five independent process runs in `benchmark_results/rtx5060_local/perf_map/`.",
        "Each process reports the median of three CUDA Event samples; the table reports medians across process medians.",
        "Protocol: 50 warmups and 200 measured launches per sample. Values are microbenchmark latency, not model-level speedup.",
        "",
        "The PyTorch native reference materializes `x + residual` and the RMSNorm output; custom `_out` variants use preallocated output buffers. Treat the ratios as an operator-path comparison, not an allocation-matched library benchmark.",
        "",
    ]

    def process_values(dtype: str, tokens: int, hidden: int, impl: str) -> list[float]:
        return grouped[(dtype, tokens, hidden, impl)]

    for dtype in ("float16", "bfloat16"):
        keys = sorted({(t, h) for d, t, h, _ in grouped if d == dtype})
        rows = []
        for tokens, hidden in keys:
            names = {impl for d, t, h, impl in grouped if (d, t, h) == (dtype, tokens, hidden)}
            native = process_values(dtype, tokens, hidden, "torch_native_fused")
            v2 = process_values(dtype, tokens, hidden, "cuda_fused_v2_out")
            selected_name = "cuda_fused_v4_packed_out" if hidden == 1024 else "cuda_fused_v2_out"
            selected = process_values(dtype, tokens, hidden, selected_name)
            row = {
                "tokens": tokens,
                "hidden": hidden,
                "torch_native_fused_median_us": statistics.median(native),
                "cuda_fused_v2_out_median_us": statistics.median(v2),
                "selected_variant": selected_name,
                "selected_median_us": statistics.median(selected),
                "selected_speedup_vs_torch_native_median_within_process": statistics.median(
                    [a / b for a, b in zip(native, selected)]
                ),
                "selected_process_latency_range_us": [min(selected), max(selected)],
            }
            if hidden == 1024:
                v3 = process_values(dtype, tokens, hidden, "cuda_fused_v3_h1024_out")
                v4 = process_values(dtype, tokens, hidden, "cuda_fused_v4_packed_out")
                row.update({
                    "cuda_fused_v3_out_median_us": statistics.median(v3),
                    "cuda_fused_v4_out_median_us": statistics.median(v4),
                    "v4_speedup_vs_v3_median_paired": statistics.median(
                        [a / b for a, b in zip(v3, v4)]
                    ),
                    "measured_implementations": sorted(names),
                })
            rows.append(row)
        summary[dtype] = rows
        md.extend([
            f"## {dtype}",
            "",
            "| Tokens | Hidden | PyTorch native (μs) | CUDA V2 out (μs) | Selected custom (μs) | Speedup vs native |",
            "|---:|---:|---:|---:|---:|---:|",
        ])
        for row in rows:
            md.append(
                f"| {row['tokens']} | {row['hidden']} | "
                f"{row['torch_native_fused_median_us']:.3f} | "
                f"{row['cuda_fused_v2_out_median_us']:.3f} | "
                f"{row['selected_median_us']:.3f} ({row['selected_variant']}) | "
                f"{row['selected_speedup_vs_torch_native_median_within_process']:.2f}× |"
            )
        md.append("")

    out_payload = {
        "gpu": environments[0]["gpu"],
        "compute_capability": environments[0]["compute_capability"],
        "torch": environments[0]["torch"],
        "cuda": environments[0]["cuda"],
        "independent_processes_per_dtype": len(next(iter(runs_by_dtype.values()))),
        "warmup_per_sample": 50,
        "iterations_per_sample": 200,
        "within_process_samples": 3,
        "results": summary,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out_payload, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
