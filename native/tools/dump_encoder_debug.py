"""Development-only block-zero fixtures used to validate Vulkan operators."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    library = torch.library.Library("torchvision", "DEF")
    library.define(
        "nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")
    sys.path.insert(0, str((args.repo / "src").resolve()))
    from depth_pro.depth_pro import create_backbone_model

    model, _ = create_backbone_model("dinov2l16_384")
    state = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True)
    prefix = "encoder.patch_encoder."
    model.load_state_dict({
        name[len(prefix):]: value
        for name, value in state.items()
        if name.startswith(prefix)
    }, strict=True)
    model.eval()
    captured: dict[str, torch.Tensor] = {}
    handles = []
    modules = {
        "norm1": model.blocks[0].norm1,
        "qkv": model.blocks[0].attn.qkv,
        "attention": model.blocks[0].attn,
        "norm2": model.blocks[0].norm2,
        "fc1": model.blocks[0].mlp.fc1,
        "mlp": model.blocks[0].mlp,
    }
    for block_index in range(6):
        modules[f"block{block_index}"] = model.blocks[block_index]
    for name, module in modules.items():
        handles.append(module.register_forward_hook(
            lambda _module, _input, output, key=name:
            captured.__setitem__(key, output.detach().cpu())))
    value = torch.from_numpy(
        np.fromfile(args.input, dtype=np.float32)
        .reshape(1, 3, 384, 384))
    with torch.inference_mode():
        model(value)
    for handle in handles:
        handle.remove()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, tensor in captured.items():
        tensor.numpy().astype(np.float32).tofile(
            args.output / f"{name}.bin")


if __name__ == "__main__":
    main()
