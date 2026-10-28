#!/usr/bin/env python3
"""Export all BF16 SmolLM3 weights from HF cache as float32 bins.

Every tensor is stored column-major. Linear weights are logical [in, out] (HF [out, in]
bytes as-is), the embedding is logical [d_model, vocab] (also the tied lm_head), and norm
vectors are logical [d, 1].

Exports: embedding table (= lm_head), all transformer blocks, final layernorm.
"""

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


# Version 2: every tensor is column-major; version 1 was row-major.
FORMAT_VERSION = 2


def linear_logical_shape(source_shape: list[int]) -> list[int]:
    """HF linear weights are [out, in]; the logical matrix for x @ W is [in, out]."""
    return [source_shape[1], source_shape[0]]


def find_safetensors()-> tuple[list[Path], Path]:
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
            shape, layout = [source_shape[0], 1], "column-major"
        else:
            if tensor.ndim != 2:
                raise ValueError(f"Expected {source_name} to be a matrix, got shape {source_shape}")
            # HF [out, in] row-major bytes are exactly the logical [in, out] matrix
            # in column-major order, so no transpose is needed.
            shape, layout = linear_logical_shape(source_shape), "column-major"

        array = tensor.to(torch.float32).contiguous().numpy()
        array = np.asarray(array, dtype=np.dtype("<f4"), order="C")
        filename = f"{output_name}.bin"
        array.tofile(output_dir / filename)

        tensors.append({
            "name": output_name,
            "file": filename,
            "shape": shape,
            "dtype": "float32",
            "byte_order": "little-endian",
            "layout": layout,
            "source_tensor": source_name,
            "source_shape": source_shape,
            "transposed_from_source": False,
            "size_bytes": array.nbytes,
        })

    manifest = {
        "format": "tinyserve-float32-matrices",
        "format_version": FORMAT_VERSION,
        "source_checkpoint": source,
        "layer_index": layer_idx,
        "tensors": tensors,
    }
    (output_dir / "shapes.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def export_single_tensor(shards, keys_by_shard: dict[int, set[str]],
                         tensor_name: str, output_name: str, kind: str,
                         output_dir: Path, source: str) -> None:
    """Export a single top-level tensor (embedding, final_norm) to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    tensor = None
    for shard_idx, keys in keys_by_shard.items():
        if tensor_name in keys:
            tensor = shards[shard_idx].get_tensor(tensor_name)
            break
    if tensor is None:
        raise ValueError(f"Missing tensor in checkpoint: {tensor_name}")

    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"Expected {tensor_name} to be BF16, got {tensor.dtype}")

    source_shape = list(tensor.shape)

    if kind == "vector":
        if tensor.ndim != 1:
            raise ValueError(f"Expected {tensor_name} to be a vector, got shape {source_shape}")
        shape, layout = [source_shape[0], 1], "column-major"
    elif kind == "matrix":
        # Embedding [vocab, d] row-major bytes == logical [d, vocab] column-major: token t is
        # column t. Same orientation as a linear weight, which is why it doubles as lm_head.
        if tensor.ndim != 2:
            raise ValueError(f"Expected {tensor_name} to be a matrix, got shape {source_shape}")
        shape, layout = linear_logical_shape(source_shape), "column-major"
    else:  # "linear" — logical [in, out] stored column-major (= HF bytes as-is)
        if tensor.ndim != 2:
            raise ValueError(f"Expected {tensor_name} to be a matrix, got shape {source_shape}")
        shape, layout = linear_logical_shape(source_shape), "column-major"

    array = tensor.to(torch.float32).contiguous().numpy()
    array = np.asarray(array, dtype=np.dtype("<f4"), order="C")
    filename = f"{output_name}.bin"
    array.tofile(output_dir / filename)

    info = {
        "name": output_name,
        "file": filename,
        "shape": shape,
        "dtype": "float32",
        "byte_order": "little-endian",
        "layout": layout,
        "source_tensor": tensor_name,
        "source_shape": source_shape,
        "transposed_from_source": False,
        "size_bytes": array.nbytes,
    }

    # For embedding, add vocab_size and embed_dim for convenience
    if output_name == "embed_tokens":
        info["vocab_size"] = source_shape[0]
        info["embed_dim"] = source_shape[1]

    manifest = {
        "format": "tinyserve-float32-matrices",
        "format_version": FORMAT_VERSION,
        "source_checkpoint": source,
        "tensor": info,
    }
    (output_dir / f"{output_name}.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


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
    print(f"Detected {num_layers} transformer layers, exporting to {output_root}/")

    source = str(index_path.parent)

    # --- Export embedding table ---
    export_single_tensor(
        opened, keys_by_shard, "model.embed_tokens.weight",
        "embed_tokens", "matrix",
        output_root, source,
    )
    print("  embed_tokens done")

    # --- Export transformer blocks ---
    for idx in range(num_layers):
        block_dir = output_root / f"bloc_{idx}"
        export_block(opened, keys_by_shard, idx, block_dir, source)
        print(f"  bloc_{idx} done")

    # --- Export final layernorm ---
    export_single_tensor(
        opened, keys_by_shard, "model.norm.weight",
        "final_norm", "vector",
        output_root, source,
    )
    print("  final_norm done")

    # lm_head is tied to embed_tokens (tie_word_embeddings=true): the runtime reuses the
    # column-major embed_tokens mapping, so no separate file is written.

    # --- Copy tokenizer files ---
    import shutil
    snapshot_dir = index_path.parent
    tokenizer_dir = output_root / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("tokenizer.json", "tokenizer_config.json"):
        src = snapshot_dir / fname
        if src.is_file():
            shutil.copy2(src, tokenizer_dir / fname)
            print(f"  {fname} copied")
        else:
            print(f"  {fname} not found in snapshot, skipping")

    print(f"Done. Embedding (= lm_head) + {num_layers} blocks + final_norm + tokenizer written to {output_root}/")


if __name__ == "__main__":
    main()
