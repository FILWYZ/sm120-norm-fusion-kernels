"""Compare nano-vLLM RMSNorm backends at larger decode batches."""
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

import torch
from nanovllm import LLM, SamplingParams


def worker(args):
    llm = LLM(
        args.model,
        attention_backend="flashinfer",
        rmsnorm_backend=args.rmsnorm_backend,
        qk_norm_rope_backend="torch",
        max_model_len=args.max_model_len,
        max_num_seqs=max(args.requests),
        max_num_batched_tokens=max(2048, args.max_model_len * max(args.requests)),
        gpu_memory_utilization=args.gpu_memory_utilization,
        gpu_memory_reserve_gib=args.gpu_memory_reserve_gib,
        enable_mixed_batching=True,
    )
    results = []
    params = SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=args.output_len)
    for count in args.requests:
        prompts = [[(17 * (i + 1) + j) % 10000 + 1 for j in range(args.input_len)] for i in range(count)]
        llm.generate(prompts, params, use_tqdm=False)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        llm.generate(prompts, params, use_tqdm=False)
        torch.cuda.synchronize()
        row = dict(llm.last_metrics)
        row.update({
            "requests": count,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        })
        results.append(row)
    payload = {
        "environment": {
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "model": args.model,
        "rmsnorm_backend": args.rmsnorm_backend,
        "qk_norm_rope_backend": "torch",
        "results": results,
    }
    args.worker_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    llm.exit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/home/asus/models/Qwen3-0.6B")
    parser.add_argument("--rmsnorm-backend", choices=["torch", "sm120"], default="sm120")
    parser.add_argument("--requests", type=int, nargs="+", default=[33, 64])
    parser.add_argument("--input-len", type=int, default=64)
    parser.add_argument("--output-len", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--gpu-memory-reserve-gib", type=float, default=0.50)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results/e2e_batch_dispatch"))
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args()
    if args.worker_output:
        worker(args)
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payloads = []
    for index in range(args.runs):
        out = args.output_dir / f"run_{index + 1:02d}.json"
        command = [
            sys.executable, str(Path(__file__).resolve()),
            "--model", args.model, "--rmsnorm-backend", args.rmsnorm_backend,
            "--requests", *map(str, args.requests),
            "--input-len", str(args.input_len), "--output-len", str(args.output_len),
            "--max-model-len", str(args.max_model_len),
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            "--gpu-memory-reserve-gib", str(args.gpu_memory_reserve_gib),
            "--worker-output", str(out),
        ]
        print(f"process {index + 1}/{args.runs}: {args.rmsnorm_backend}", flush=True)
        subprocess.run(command, check=True)
        payloads.append(json.loads(out.read_text(encoding="utf-8")))
    workloads = []
    for index, count in enumerate(args.requests):
        rows = [payload["results"][index] for payload in payloads]
        metrics = {}
        for name in rows[0]:
            values = [row[name] for row in rows]
            if all(isinstance(value, (int, float)) for value in values):
                metrics[name] = {
                    "median": statistics.median(values),
                    "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "runs": values,
                }
        workloads.append({"requests": count, "metrics": metrics})
    summary = {
        "schema_version": 1,
        "method": "independent_processes",
        "runs": args.runs,
        "rmsnorm_backend": args.rmsnorm_backend,
        "environment": payloads[-1]["environment"],
        "config": {"model": args.model, "input_len": args.input_len, "output_len": args.output_len,
                   "max_model_len": args.max_model_len, "qk_norm_rope_backend": "torch"},
        "workloads": workloads,
    }
    out = args.output_dir / "summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"summary={out}")


if __name__ == "__main__":
    main()
