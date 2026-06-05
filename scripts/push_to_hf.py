#!/usr/bin/env python3
"""
Push a directory (e.g. the merged MaCroScope-Mathematics LoRA model) to a
Hugging Face repo under the ai4collaboration org.

Usage:
    HF_TOKEN=hf_xxx python push_to_hf.py \
        --local-dir   /path/to/merged-final \
        --repo-id     ai4collaboration/MaCroScope-Mathematics-Nemotron-30B-LoRA-merged \
        --repo-type   model \
        --private               # optional
        --commit-message "Initial upload" \
        --allow-patterns "*"    # optional, restrict what gets uploaded
"""
import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-dir", required=True, help="Local directory to upload.")
    ap.add_argument("--repo-id", required=True,
                    help="Target HF repo id, e.g. ai4collaboration/MaCroScope-Mathematics-Nemotron-30B-LoRA-merged")
    ap.add_argument("--repo-type", default="model", choices=["model", "dataset"])
    ap.add_argument("--private", action="store_true",
                    help="Create the repo as private (no-op if it already exists).")
    ap.add_argument("--commit-message", default="Upload from Fir compute job")
    ap.add_argument("--allow-patterns", nargs="*", default=None,
                    help="Whitelist glob(s); default = all files in local-dir.")
    ap.add_argument("--ignore-patterns", nargs="*",
                    default=["*.tmp", "*.lock", "__pycache__/*", ".DS_Store"],
                    help="Globs to skip.")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token or token == "REPLACE_ME":
        print("ERROR: HF_TOKEN env var is not set (or is placeholder).", file=sys.stderr)
        return 2

    local = Path(args.local_dir)
    if not local.is_dir():
        print(f"ERROR: local-dir not found: {local}", file=sys.stderr)
        return 2

    api = HfApi(token=token)

    print(f"[hf] creating repo if missing: {args.repo_id} (type={args.repo_type}, private={args.private})")
    create_repo(
        repo_id=args.repo_id,
        token=token,
        repo_type=args.repo_type,
        private=args.private,
        exist_ok=True,
    )

    print(f"[hf] uploading folder {local} -> {args.repo_id}")
    info = api.upload_folder(
        folder_path=str(local),
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        commit_message=args.commit_message,
        allow_patterns=args.allow_patterns,
        ignore_patterns=args.ignore_patterns,
    )
    print(f"[hf] done: {info}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
