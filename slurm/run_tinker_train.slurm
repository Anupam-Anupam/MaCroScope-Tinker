#!/bin/bash
#SBATCH --job-name=tinker-nemotron-biochem23k
#SBATCH --account=rrg-zhijing
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
date

if [ -f "$HOME/.tinker_env" ]; then
    # shellcheck disable=SC1091
    source "$HOME/.tinker_env"
fi
if [ -z "${TINKER_API_KEY:-}" ] || [ "${TINKER_API_KEY}" = "REPLACE_ME" ]; then
    echo "ERROR: TINKER_API_KEY not set. Edit ~/.tinker_env (chmod 600)." >&2
    exit 1
fi

module load python/3.11 cuda/12.2 gcc opencv arrow
source ~/envs/macroscope/bin/activate

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT"

LOG_DIR="${LOG_DIR:-./log_nemotron_biochem23k_grstyle}"
mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR/checkpoints.jsonl"

echo "=== Tinker GRPO: Nemotron-30B + MaCroScope-Biology-Chemistry-23k ==="
echo "Repo: $REPO_ROOT"
echo "Log dir: $LOG_DIR"

python scripts/tinker_train_general_reasoner.py \
    log_path="$LOG_DIR"

echo "=== Job Complete ==="
date
