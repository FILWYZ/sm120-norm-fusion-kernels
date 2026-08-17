#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

ncu \
  --set full \
  --kernel-name regex:.*fused_add_rms_norm_h1024_kernel.* \
  --launch-skip 100 \
  --launch-count 1 \
  --export benchmark_results/ncu_v3_h1024 \
  "${PYTHON_BIN}" -m benchmarks.profile_case \
    --tokens 64 \
    --hidden 1024 \
    --version 3 \
    --warmup 100
