#!/usr/bin/env bash
# One-time MaCroScope Tinker env setup on CSCS Clariden (ARM64 GH200).
# Run from MaCroScope-Tinker repo root after loading your container/uenv.
#
# Swiss AI onboarding: https://github.com/swiss-ai/reasoning_getting-started
# CSCS docs: https://docs.cscs.ch/clusters/clariden
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_DIR="${ENV_DIR:-$HOME/envs/macroscope}"
TINKER_COOKBOOK_DIR="${TINKER_COOKBOOK_DIR:-$HOME/scratch/tinker-cookbook}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/iopsstor/scratch/cscs/${USER}}"

mkdir -p "$SCRATCH_ROOT"

echo "=== MaCroScope Tinker setup (Clariden) ==="
echo "REPO_ROOT=$REPO_ROOT"
echo "ENV_DIR=$ENV_DIR"
echo "TINKER_COOKBOOK_DIR=$TINKER_COOKBOOK_DIR"

# Caches on scratch (not 50GB /users home)
export HF_HOME="${HF_HOME:-$SCRATCH_ROOT/huggingface_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"

if [ -f "$HOME/user.env" ]; then
  # shellcheck disable=SC1091
  source "$HOME/user.env"
elif [ -f "$HOME/.tinker_env" ]; then
  # shellcheck disable=SC1091
  source "$HOME/.tinker_env"
fi

python3 -m venv "$ENV_DIR"
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"

pip install --upgrade pip wheel setuptools
pip install -r "$REPO_ROOT/clariden/requirements-clariden.txt"

if [ ! -d "$TINKER_COOKBOOK_DIR/.git" ]; then
  git clone https://github.com/thinking-machines-lab/tinker-cookbook.git "$TINKER_COOKBOOK_DIR"
fi
pip install -e "$TINKER_COOKBOOK_DIR"

cat >> "$HOME/.bashrc" <<'RC'

# MaCroScope Tinker (Clariden)
export HF_HOME="/iopsstor/scratch/cscs/${USER}/huggingface_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
RC

echo ""
echo "=== Done ==="
echo "Activate: source $ENV_DIR/bin/activate"
echo "Secrets: cp clariden/user.env.example ~/user.env && chmod 600 ~/user.env"
echo "Test: python -c \"import tinker, vllm, transformers; print('ok')\""
