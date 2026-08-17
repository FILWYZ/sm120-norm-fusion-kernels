import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=["fp16", "bf16"], required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results/repeated"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_payloads = []
    for run in range(args.repeats):
        output = args.output_dir / f"{args.dtype}_run_{run + 1}.json"
        command = [
            sys.executable,
            "-m",
            "benchmarks.benchmark",
            "--tokens",
            "1",
            "16",
            "64",
            "256",
            "--hidden",
            "1024",
            "--dtype",
            args.dtype,
            "--warmup",
            str(args.warmup),
            "--iterations",
            str(args.iterations),
            "--output",
            str(output),
        ]
        print(f"[{run + 1}/{args.repeats}] {' '.join(command)}", flush=True)
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        run_payloads.append(json.loads(output.read_text(encoding="utf-8")))

    grouped: dict[tuple[str, int, int, str], list[float]] = {}
    for payload in run_payloads:
        for row in payload["results"]:
            key = (
                row["implementation"],
                row["tokens"],
                row["hidden"],
                row["dtype"],
            )
            grouped.setdefault(key, []).append(row["latency_us"])

    summary = []
    for (implementation, tokens, hidden, dtype), values in sorted(grouped.items()):
        median = statistics.median(values)
        summary.append(
            {
                "implementation": implementation,
                "tokens": tokens,
                "hidden": hidden,
                "dtype": dtype,
                "median_latency_us": median,
                "min_latency_us": min(values),
                "max_latency_us": max(values),
                "relative_range_pct": (max(values) - min(values)) / median * 100.0,
                "independent_run_latencies_us": values,
            }
        )

    aggregate = {
        "environment": run_payloads[0]["environment"],
        "protocol": {
            "independent_processes": args.repeats,
            "warmup_per_sample": args.warmup,
            "iterations_per_sample": args.iterations,
            "within_process_samples": 3,
        },
        "results": summary,
    }
    aggregate_path = args.output_dir / f"{args.dtype}_aggregate.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(f"wrote {aggregate_path}")


if __name__ == "__main__":
    main()

