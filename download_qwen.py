#!/usr/bin/env python3
"""Download a Qwen model from the Hugging Face Hub and save its weights as FP16 safetensors.

Usage:
    python download_qwen.py                          # Qwen/Qwen3-0.6B -> models/Qwen3-0.6B-fp16
    python download_qwen.py --repo Qwen/Qwen2.5-1.5B-Instruct --out models/qwen2.5-1.5b-fp16

Requires: pip install huggingface_hub safetensors torch
"""

import argparse
import shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", default="Qwen/Qwen3-0.6B", help="Hugging Face repo id")
    p.add_argument("--out", default=None, help="output directory (default: models/<name>-fp16)")
    p.add_argument("--revision", default=None, help="branch/tag/commit to download")
    args = p.parse_args()

    out = Path(args.out or f"models/{args.repo.split('/')[-1]}-fp16")
    out.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.repo} ...")
    src = Path(
        snapshot_download(
            repo_id=args.repo,
            revision=args.revision,
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model", "*.tiktoken"],
        )
    )

    # Copy config/tokenizer files as-is.
    for f in src.iterdir():
        if f.suffix != ".safetensors" and not f.name.endswith(".safetensors.index.json"):
            shutil.copy(f, out / f.name)

    # Convert every weight shard to FP16 (Qwen ships BF16 by default).
    shards = sorted(src.glob("*.safetensors"))
    if not shards:
        raise SystemExit("no .safetensors files found in repo")
    for shard in shards:
        tensors = load_file(shard)
        fp16 = {
            k: (v.to(torch.float16) if v.is_floating_point() else v).contiguous()
            for k, v in tensors.items()
        }
        save_file(fp16, out / shard.name, metadata={"format": "pt"})
        print(f"  {shard.name}: {len(fp16)} tensors -> fp16")

    index = src / "model.safetensors.index.json"
    if index.exists():
        shutil.copy(index, out / index.name)

    print(f"Done: {out.resolve()}")


if __name__ == "__main__":
    main()
