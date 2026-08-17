import argparse
import json
from pathlib import Path


IMPLEMENTATIONS = [
    "torch_native_fused",
    "cuda_fused_v2_out",
    "cuda_fused_v3_h1024_out",
    "cuda_fused_v4_packed_out",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    rows = payload["results"]
    tokens = sorted({row["tokens"] for row in rows})

    print("| Tokens | PyTorch native | CUDA V2 | CUDA V3 | CUDA V4 | Best speedup |")
    print("|---:|---:|---:|---:|---:|---:|")
    for token_count in tokens:
        by_name = {
            row["implementation"]: row["latency_us"]
            for row in rows
            if row["tokens"] == token_count
        }
        values = [by_name[name] for name in IMPLEMENTATIONS]
        best = min(values[1:])
        speedup = values[0] / best
        print(
            f"| {token_count} | {values[0]:.3f} μs | {values[1]:.3f} μs | "
            f"{values[2]:.3f} μs | {values[3]:.3f} μs | {speedup:.2f}× |"
        )


if __name__ == "__main__":
    main()

