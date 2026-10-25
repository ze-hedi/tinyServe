#!/usr/bin/env python3
"""Export all BF16 SmolLM3 transformer blocks from HF cache as row-major float32 bins."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

MODEL_REPO = "models--HuggingFaceTB--SmolLM3-3B"
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"

TENSORS = (
    ("input_layernorm", "input_layernorm.weight", "vector"),
    ("q_proj", "self_attn.q_proj.weight", "linear"),
    ("k_proj", "self_attn.k_proj.weight", "linear"),
    ("v_proj", "self_attn.v_proj.weight", "linear"),
    ("o_proj", "self_attn.o_proj.weight", "linear"),
    ("post_attention_layernorm", "post_attention_layernorm.weight", "vector"),
    ("gate_proj", "mlp.gate_proj.weight", "linear"),
    ("up_proj", "mlp.up_proj.weight", "linear"),
    ("down_proj", "mlp.down_proj.weight", "linear"),
)


def find_safetensors() -> tuple[list[Path], Path]:
    """Locate the SmolLM3-3B safetensors shards and index in HF cache."""
    model_dir = HF_CACHE / MODEL_REPO
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model not found in HF cache: {model_dir}")

    snapshots = model_dir / "snapshots"
    # Pick the most recent snapshot (usually only one)
    revisions = sorted(snapshots.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    if not revisions:
        raise FileNotFoundError(f"No snapshots found in {snapshots}")
    snapshot = revisions[0]

    index_path = snapshot / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Index file not found: {index_path}")

    # Collect unique shard filenames from the index
    index = json.loads(index_path.read_text())
    shard_names = sorted(set(index["weight_map"].values()))
    shards = []
    for name in shard_names:
        path = snapshot / name
        if not path.exists():
            raise FileNotFoundError(f"Shard not found: {path}")
        shards.append(path)

    return shards, index_path


def detect_layer_count(keys: set[str]) -> int:
    indices = set()
    for key in keys:
        m = re.match(r"model\.layers\.(\d+)\.", key)
        if m:
            indices.add(int(m.group(1)))
    if not indices:
        raise ValueError("No model.layers.* tensors found in checkpoint")
    return max(indices) + 1


def export_block(shards, keys_by_shard: dict[int, set[str]], layer_idx: int,
                 output_dir: Path, source: str) -> None:
    prefix = f"model.layers.{layer_idx}."
    output_dir.mkdir(parents=True, exist_ok=True)
    tensors = []

    for output_name, suffix, kind in TENSORS:
        source_name = prefix + suffix

        # Find which shard has this tensor
        tensor = None
        for shard_idx, keys in keys_by_shard.items():
            if source_name in keys:
                tensor = shards[shard_idx].get_tensor(source_name)
                break
        if tensor is None:
            raise ValueError(f"Missing tensor in checkpoint: {source_name}")

        if tensor.dtype != torch.bfloat16:
            raise ValueError(f"Expected {source_name} to be BF16, got {tensor.dtype}")

        source_shape = list(tensor.shape)
        if kind == "vector":
            if tensor.ndim != 1:
                raise ValueError(f"Expected {source_name} to be a vector, got shape {source_shape}")
            tensor = tensor.reshape(1, -1)
            transposed = False
        else:
            if tensor.ndim != 2:
                raise ValueError(f"Expected {source_name} to be a matrix, got shape {source_shape}")
            tensor = tensor.transpose(0, 1)
            transposed = True

        array = tensor.to(torch.float32).contiguous().numpy()
        array = np.asarray(array, dtype=np.dtype("<f4"), order="C")
        filename = f"{output_name}.bin"
        array.tofile(output_dir / filename)

        tensors.append({
            "name": output_name,
            "file": filename,
            "shape": list(array.shape),
            "dtype": "float32",
            "byte_order": "little-endian",
            "layout": "row-major",
            "source_tensor": source_name,
            "source_shape": source_shape,
            "transposed_from_source": transposed,
            "size_bytes": array.nbytes,
        })

    manifest = {
        "format": "tinyserve-float32-matrices",
        "format_version": 1,
        "source_checkpoint": source,
        "layer_index": layer_idx,
        "tensors": tensors,
    }
    (output_dir / "shapes.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    shard_paths, index_path = find_safetensors()
    print(f"Found {len(shard_paths)} shard(s):")
    for p in shard_paths:
        print(f"  {p}")

    # Open all shards and collect their keys
    opened = []
    keys_by_shard: dict[int, set[str]] = {}
    all_keys: set[str] = set()
    for i, path in enumerate(shard_paths):
        f = safe_open(str(path), framework="pt", device="cpu")
        opened.append(f)
        k = set(f.keys())
        keys_by_shard[i] = k
        all_keys |= k

    num_layers = detect_layer_count(all_keys)
    output_root = Path("smollm3")
    print(f"Detected {num_layers} transformer layers, exporting all blocks to {output_root}/")

    source = str(index_path.parent)
    for idx in range(num_layers):
        block_dir = output_root / f"bloc_{idx}"
        export_block(opened, keys_by_shard, idx, block_dir, source)
        print(f"  bloc_{idx} done")

    print(f"Done. {num_layers} blocks written to {output_root}/")


if __name__ == "__main__":
    main()
