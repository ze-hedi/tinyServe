#!/usr/bin/env python3
"""Export the first BF16 Llama block from safetensors as row-major float32 bins."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


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


def export_first_block(checkpoint: Path, output_dir: Path) -> None:
    prefix = "model.layers.0."
    output_dir.mkdir(parents=True, exist_ok=True)
    tensors = []

    with safe_open(str(checkpoint), framework="pt", device="cpu") as file:
        keys = set(file.keys())
        for output_name, suffix, kind in TENSORS:
            source_name = prefix + suffix
            if source_name not in keys:
                raise ValueError(f"Missing tensor in checkpoint: {source_name}")

            tensor = file.get_tensor(source_name)
            if tensor.dtype != torch.bfloat16:
                raise ValueError(
                    f"Expected {source_name} to be BF16, got {tensor.dtype}"
                )

            source_shape = list(tensor.shape)
            if kind == "vector":
                if tensor.ndim != 1:
                    raise ValueError(
                        f"Expected {source_name} to be a vector, got shape {source_shape}"
                    )
                # The C++ loader represents normalization vectors as (1, d_model).
                tensor = tensor.reshape(1, -1)
                transposed = False
            else:
                if tensor.ndim != 2:
                    raise ValueError(
                        f"Expected {source_name} to be a matrix, got shape {source_shape}"
                    )
                # HF linear weights are [out, in]; Mat weights are [in, out].
                tensor = tensor.transpose(0, 1)
                transposed = True

            array = tensor.to(torch.float32).contiguous().numpy()
            array = np.asarray(array, dtype=np.dtype("<f4"), order="C")
            filename = f"{output_name}.bin"
            array.tofile(output_dir / filename)

            tensors.append(
                {
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
                }
            )

    manifest = {
        "format": "tinyserve-float32-matrices",
        "format_version": 1,
        "source_checkpoint": str(checkpoint),
        "layer_index": 0,
        "tensors": tensors,
    }
    (output_dir / "shapes.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the first BF16 Llama block in a safetensors checkpoint "
            "to row-major float32 binary matrices."
        )
    )
    parser.add_argument("checkpoint", type=Path, help="Input .safetensors file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("bloc_0"),
        help="Output directory (default: ./bloc_0)",
    )
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        parser.error(f"checkpoint file does not exist: {args.checkpoint}")

    export_first_block(args.checkpoint, args.output_dir)
    print(f"Exported first block to {args.output_dir}")


if __name__ == "__main__":
    main()
