#!/usr/bin/env bash
set -euo pipefail

CUDA_DISASM_BIN="${CUDA_DISASM_BIN:-/usr/local/cuda-13.3/bin}"
EXTENSION_PATH="${1:-sm120_rmsnorm/_C.cpython-312-x86_64-linux-gnu.so}"

if [[ ! -x "${CUDA_DISASM_BIN}/cuobjdump" || ! -x "${CUDA_DISASM_BIN}/nvdisasm" ]]; then
  echo "cuobjdump and nvdisasm are required under ${CUDA_DISASM_BIN}" >&2
  exit 1
fi

if [[ ! -f "${EXTENSION_PATH}" ]]; then
  echo "extension not found: ${EXTENSION_PATH}" >&2
  exit 1
fi

SASS_FILE="$(mktemp)"
trap 'rm -f "${SASS_FILE}"' EXIT

PATH="${CUDA_DISASM_BIN}:${PATH}" \
NVDISASM_PATH="${CUDA_DISASM_BIN}/nvdisasm" \
  "${CUDA_DISASM_BIN}/cuobjdump" --dump-sass "${EXTENSION_PATH}" >"${SASS_FILE}"

LOADS="$(grep -c 'LDG.E.128' "${SASS_FILE}" || true)"
STORES="$(grep -c 'STG.E.128' "${SASS_FILE}" || true)"

if [[ "${LOADS}" -eq 0 || "${STORES}" -eq 0 ]]; then
  echo "128-bit global-memory instructions were not found" >&2
  exit 1
fi

echo "verified ${LOADS} LDG.E.128 and ${STORES} STG.E.128 instructions"
grep -n -m 8 -E 'LDG.E.128|STG.E.128' "${SASS_FILE}"
