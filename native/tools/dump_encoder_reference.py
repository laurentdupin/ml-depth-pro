"""Generate deterministic Depth Pro patch-encoder fixtures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str((args.repo / "src").resolve()))
    from depth_pro.depth_pro import create_backbone_model

    model, _ = create_backbone_model("dinov2l16_384")
    state = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True)
    prefix = "encoder.patch_encoder."
    patch_state = {
        name[len(prefix):]: value
        for name, value in state.items()
        if name.startswith(prefix)
    }
    model.load_state_dict(patch_state, strict=True)
    model.eval()
    captures = {}
    handles = []
    for block_index in (5, 11, 17, 23):
        handles.append(model.blocks[block_index].register_forward_hook(
            lambda _module, _input, output, index=block_index:
            captures.__setitem__(index, output.detach().cpu())))
    generator = torch.Generator().manual_seed(20260730)
    value = torch.rand(1, 3, 384, 384, generator=generator)
    value = (value - 0.5) / 0.5
    with torch.inference_mode():
        model(value)
    for handle in handles:
        handle.remove()

    output_prefix = args.output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    value.numpy().astype(np.float32).tofile(
        output_prefix.with_suffix(".input.bin"))
    report = {"input_shape": list(value.shape), "captures": []}
    for block_index in (5, 11, 17, 23):
        tensor = captures[block_index]
        tensor.numpy().astype(np.float32).tofile(
            output_prefix.parent /
            f"{output_prefix.name}.block{block_index}.bin")
        report["captures"].append({
            "block": block_index,
            "shape": list(tensor.shape),
            "minimum": float(tensor.min()),
            "maximum": float(tensor.max()),
            "mean": float(tensor.mean()),
            "sum": float(tensor.double().sum()),
        })
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
