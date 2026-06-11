#!/usr/bin/env bash
# Run on Fir (Alliance) to bundle portable artifacts for Clariden migration.
# Does NOT include secrets or the Fir venv (venv is x86_64 + CVMFS — recreate on Clariden).
set -euo pipefail

SCRATCH="${SCRATCH:-/scratch/anupam}"
BUNDLE_DIR="${BUNDLE_DIR:-$SCRATCH/clariden_transport_bundle}"
TARBALL="${TARBALL:-$SCRATCH/clariden_transport_bundle.tar.gz}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

rm -rf "$BUNDLE_DIR"
mkdir -p "$BUNDLE_DIR"

echo "=== Exporting MaCroScope-Tinker repo ==="
rsync -a --exclude '.git' "$REPO_ROOT/" "$BUNDLE_DIR/MaCroScope-Tinker/"

echo "=== Exporting tinker-cookbook ==="
if [ -d "$SCRATCH/tinker-cookbook/.git" ]; then
  rsync -a --exclude '.git' "$SCRATCH/tinker-cookbook/" "$BUNDLE_DIR/tinker-cookbook/"
else
  echo "WARN: $SCRATCH/tinker-cookbook not found; clone on Clariden instead."
fi

echo "=== Exporting pip freeze from Fir env ==="
if [ -f "$HOME/envs/macroscope/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$HOME/envs/macroscope/bin/activate"
  pip freeze > "$BUNDLE_DIR/requirements-fir-freeze.txt"
  sed 's/+computecanada//g' "$BUNDLE_DIR/requirements-fir-freeze.txt" \
    > "$BUNDLE_DIR/requirements-fir-freeze-sanitized.txt"
fi

echo "=== Copying env template (no secrets) ==="
cp "$REPO_ROOT/clariden/user.env.example" "$BUNDLE_DIR/user.env.example"

cat > "$BUNDLE_DIR/README_TRANSFER.txt" <<EOF
MaCroScope Fir -> Clariden transport bundle
Created: $(date -Iseconds) on $(hostname)

Transfer to Clariden:
  scp $TARBALL clariden:~/scratch/

On Clariden:
  tar -xzf ~/scratch/clariden_transport_bundle.tar.gz -C ~/scratch/
  cd ~/scratch/clariden_transport_bundle
  bash MaCroScope-Tinker/clariden/setup_clariden.sh

Or clone from GitHub instead of using bundled repo:
  git clone https://github.com/Anupam-Anupam/MaCroScope-Tinker.git
EOF

tar -czf "$TARBALL" -C "$(dirname "$BUNDLE_DIR")" "$(basename "$BUNDLE_DIR")"
echo "=== Done: $TARBALL ($(du -h "$TARBALL" | awk '{print $1}')) ==="
