#!/usr/bin/env bash
# Run on Clariden login node (inside container if needed).
# Downloads transport bundle + secrets from private HF repos (no scp from Fir required).
set -euo pipefail

SCRATCH_ROOT="${SCRATCH_ROOT:-$HOME/scratch}"
mkdir -p "$SCRATCH_ROOT"

if [ -f "$HOME/user.env" ]; then
  # shellcheck disable=SC1091
  source "$HOME/user.env"
elif [ -f "$HOME/.tinker_env" ]; then
  # shellcheck disable=SC1091
  source "$HOME/.tinker_env"
fi

if [ -z "${HF_TOKEN:-}" ] || [ "${HF_TOKEN}" = "REPLACE_ME" ]; then
  echo "ERROR: HF_TOKEN required to download private migration repos." >&2
  echo "Add HF_TOKEN to ~/user.env first (one-time manual paste), then re-run." >&2
  exit 1
fi

source "${HOME}/envs/macroscope/bin/activate" 2>/dev/null || true
python -c "import huggingface_hub" 2>/dev/null || pip install -q huggingface_hub

python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

scratch = Path(os.environ.get("SCRATCH_ROOT", os.path.expanduser("~/scratch")))
scratch.mkdir(parents=True, exist_ok=True)

bundle = hf_hub_download(
    repo_id="ai4collaboration/clariden-transport-bundle",
    filename="clariden_transport_bundle.tar.gz",
    repo_type="model",
    local_dir=scratch,
)
print(f"[hf] bundle -> {bundle}")

secrets = hf_hub_download(
    repo_id="ai4collaboration/clariden-migration-secrets",
    filename="user.env",
    repo_type="model",
    local_dir=scratch,
)
dest = Path.home() / "user.env"
Path(secrets).replace(dest)
dest.chmod(0o600)
print(f"[hf] user.env -> {dest}")
PY

tar -xzf "$SCRATCH_ROOT/clariden_transport_bundle.tar.gz" -C "$SCRATCH_ROOT"
echo "=== Unpacked to $SCRATCH_ROOT/clariden_transport_bundle ==="
echo "Next: cd $SCRATCH_ROOT/clariden_transport_bundle/MaCroScope-Tinker && bash clariden/setup_clariden.sh"
