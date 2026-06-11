#!/bin/bash
#SBATCH --job-name=tinker-biochem23k
#SBATCH --account=YOUR_PROJECT_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --time=12:00:00
#SBATCH --partition=normal
#SBATCH --environment=my_env
#SBATCH --output=/iopsstor/scratch/cscs/%u/tinker-biochem23k-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/tinker-biochem23k-%j.err

# Clariden driver job for MaCroScope Tinker training.
# Requires EDF environment `my_env` with PyTorch + CUDA (see clariden/MIGRATION.md).
# Replace YOUR_PROJECT_ACCOUNT with your CSCS project (e.g. infra01).

set -euo pipefail

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
date

if [ -f "$HOME/user.env" ]; then
  # shellcheck disable=SC1091
  source "$HOME/user.env"
elif [ -f "$HOME/.tinker_env" ]; then
  # shellcheck disable=SC1091
  source "$HOME/.tinker_env"
fi

if [ -z "${TINKER_API_KEY:-}" ] || [ "${TINKER_API_KEY}" = "REPLACE_ME" ]; then
  echo "ERROR: TINKER_API_KEY not set in ~/user.env or ~/.tinker_env" >&2
  exit 1
fi

export HF_HOME="${HF_HOME:-/iopsstor/scratch/cscs/${USER}/huggingface_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
mkdir -p "$HF_HUB_CACHE"

source "$HOME/envs/macroscope/bin/activate"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$HOME/scratch/MaCroScope-Tinker}"
cd "$REPO_ROOT"

LOG_DIR="${LOG_DIR:-$HOME/scratch/log_nemotron_biochem23k_grstyle}"
mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR/checkpoints.jsonl"

echo "=== Tinker GRPO: Nemotron-30B + MaCroScope-Biology-Chemistry-23k ==="
echo "Repo: $REPO_ROOT"
echo "Log dir: $LOG_DIR"

python scripts/tinker_train_general_reasoner.py \
    log_path="$LOG_DIR"

echo "=== Job Complete ==="
date
