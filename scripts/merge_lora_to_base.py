#!/usr/bin/env python3
"""
Merge a Tinker-downloaded LoRA adapter into its base model and save a
self-contained HF model directory that vLLM can load directly.

The downloaded adapter (sampler_weights/...) has `base_model_name_or_path: null`
in adapter_config.json, so the base model id must be passed explicitly.
"""
import argparse
import os
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True,
                    help="HF model id or local path of the base model the LoRA was trained on.")
    ap.add_argument("--adapter", required=True,
                    help="Local path to the downloaded LoRA adapter directory (with adapter_model.safetensors).")
    ap.add_argument("--out", required=True,
                    help="Output directory for the merged model.")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device-map", default="cpu",
                    help="device_map for loading the base model. 'cpu' is safest for 30B; "
                         "use 'auto' if you have enough GPU memory.")
    ap.add_argument("--trust-remote-code", action="store_true", default=True)
    args = ap.parse_args()

    out = Path(args.out)
    if (out / "config.json").exists():
        print(f"[merge] {out}/config.json already exists; skipping merge.")
        return 0
    out.mkdir(parents=True, exist_ok=True)

    dtype = {"bfloat16": torch.bfloat16,
             "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    print(f"[merge] base={args.base}")
    print(f"[merge] adapter={args.adapter}")
    print(f"[merge] out={args.out}")
    print(f"[merge] dtype={args.dtype} device_map={args.device_map} trust_remote_code={args.trust_remote_code}")

    print("[merge] Loading base model (this can take a while for 30B)...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
    )

    print("[merge] Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=args.trust_remote_code)

    print("[merge] Attaching LoRA adapter...")
    model = PeftModel.from_pretrained(base, args.adapter)

    print("[merge] Merging LoRA weights into base (merge_and_unload)...")
    model = model.merge_and_unload()

    print(f"[merge] Saving merged model to {args.out} ...")
    model.save_pretrained(args.out, safe_serialization=True, max_shard_size="5GB")
    tok.save_pretrained(args.out)
    print("[merge] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
