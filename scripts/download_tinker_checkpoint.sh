#!/usr/bin/env bash
# Download Tinker sampler-weight checkpoints for a completed run.
#
# Usage:
#   export TINKER_RUN_ID="1591a9f4-cf08-5328-939a-20e6a9bf7c15:train:0"
#   bash scripts/download_tinker_checkpoint.sh final
#   bash scripts/download_tinker_checkpoint.sh val_000480
#
# Requires: tinker CLI (from `pip install tinker`) and TINKER_API_KEY in the environment.
set -uo pipefail

RUN_ID="${TINKER_RUN_ID:?Set TINKER_RUN_ID, e.g. 1591a9f4-cf08-5328-939a-20e6a9bf7c15:train:0}"
CHECKPOINT="${1:-final}"   # e.g. final, val_000480
OUT_DIR="${OUT_DIR:-./checkpoints}"
RUN="tinker://${RUN_ID}"

mkdir -p "$OUT_DIR"

if [ -f "$HOME/.tinker_env" ]; then
  # shellcheck disable=SC1091
  source "$HOME/.tinker_env"
fi

TARGET="sampler_weights/${CHECKPOINT}"
TARGET_NAME="${RUN_ID}_${TARGET//\//_}"

if [ -f "${OUT_DIR}/${TARGET_NAME}/checkpoint_complete" ]; then
  echo "=== SKIP ${TARGET}: already at ${OUT_DIR}/${TARGET_NAME} ==="
  exit 0
fi

echo "=== Downloading ${RUN}/${TARGET} -> ${OUT_DIR} ==="
tinker checkpoint download "${RUN}/${TARGET}" --output "${OUT_DIR}" --force
