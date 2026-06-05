#!/usr/bin/env bash
# One-time environment setup for MaCroScope Tinker training.
# Tested on Alliance Canada (Fir) with module python/3.11 cuda/12.2 gcc opencv arrow.
set -euo pipefail

ENV_DIR="${ENV_DIR:-$HOME/envs/macroscope}"
TINKER_COOKBOOK_DIR="${TINKER_COOKBOOK_DIR:-$SCRATCH/tinker-cookbook}"

echo "=== MaCroScope-Tinker env setup ==="
echo "ENV_DIR=$ENV_DIR"
echo "TINKER_COOKBOOK_DIR=$TINKER_COOKBOOK_DIR"

if command -v module >/dev/null 2>&1; then
  module load python/3.11 cuda/12.2 gcc opencv arrow
fi

python -m venv --system-site-packages "$ENV_DIR"
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"

pip install --upgrade pip wheel setuptools

# PyTorch: on Alliance, prefer the cluster-provided wheel via pip + CVMFS.
# On other machines, install CUDA-matched torch from https://pytorch.org
pip install -r requirements.txt

if [ ! -d "$TINKER_COOKBOOK_DIR/.git" ]; then
  echo "Cloning tinker-cookbook into $TINKER_COOKBOOK_DIR ..."
  git clone https://github.com/thinking-machines-lab/tinker-cookbook.git "$TINKER_COOKBOOK_DIR"
fi

pip install -e "$TINKER_COOKBOOK_DIR"

echo ""
echo "=== Done ==="
echo "Activate: source $ENV_DIR/bin/activate"
echo "Set secrets: cp .env.example ~/.tinker_env && chmod 600 ~/.tinker_env"
