"""Run alternating nano-vLLM E2E A/B tests for Q/K Norm+RoPE dispatch."""
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/home/asus/models/Qwen3-0.6B")
    parser.add_argument("--requests", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--input-len", type=int, default=64)
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=128)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--gpu-memory-reserve-gib", type=float, default=0.50)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    backends = {"torch": [], "sm120": []}
    worker = Path(__file__).with_name("e2e_batch_dispatch.py").resolve()
    for index in range(args.runs):
        order = ("torch", "sm120") if index % 2 == 0 else ("sm120", "torch")
        for backend in order:
            output = args.output_dir / f"{backend}_run_{index + 1:02d}.json"
            command = [
                sys.executable, str(worker), "--model", args.model,
                "--rmsnorm-backend", "sm120", "--qk-norm-rope-backend", backend,
                "--requests", *map(str, args.requests),
                "--input-len", str(args.input_len), "--output-len", str(args.output_len),
                "--max-model-len", str(args.max_model_len),
                "--gpu-memory-utilization", str(args.gpu_memory_utilization),
                "--gpu-memory-reserve-gib", str(args.gpu_memory_reserve_gib),
                "--worker-output", str(output),
            ]
            print(f"pair {index + 1}/{args.runs}: qk={backend}", flush=True)
            subprocess.run(command, check=True)
            backends[backend].append(json.loads(output.read_text(encoding="utf-8")))

    summary = {
        "schema_version": 1,
        "method": "alternating_independent_process_pairs",
        "runs": args.runs,
        "config": {"model": args.model, "requests": args.requests,
                   "input_len": args.input_len, "output_len": args.output_len,
                   "max_model_len": args.max_model_len,
                   "rmsnorm_backend": "sm120"},
        "environment": backends["torch"][0]["environment"],
        "workloads": [],
    }
    for wi, count in enumerate(args.requests):
        row = {"requests": count, "backends": {}, "paired_change_pct": {}}
        metrics = ("output_tokens_per_second", "decode_tokens_per_second",
                   "mean_tpot_seconds", "batch_ttft_seconds")
        for backend in ("torch", "sm120"):
            per_metric = {}
            for metric in metrics:
                values = [run["results"][wi][metric] for run in backends[backend]]
                per_metric[metric] = {"median": statistics.median(values), "runs": values}
            row["backends"][backend] = per_metric
        for metric in metrics:
            base = row["backends"]["torch"][metric]["runs"]
            fused = row["backends"]["sm120"][metric]["runs"]
            changes = [(b / a - 1.0) * 100.0 if a else 0.0 for a, b in zip(base, fused)]
            row["paired_change_pct"][metric] = {
                "median": statistics.median(changes), "per_pair": changes
            }
        summary["workloads"].append(row)
        out_gain = row["paired_change_pct"]["output_tokens_per_second"]["median"]
        dec_gain = row["paired_change_pct"]["decode_tokens_per_second"]["median"]
        print(f"requests={count} qk_sm120 output={out_gain:+.2f}% decode={dec_gain:+.2f}%", flush=True)
    path = args.output_dir / "paired_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"summary={path}")


if __name__ == "__main__":
    main()
