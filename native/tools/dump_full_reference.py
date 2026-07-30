"""Generate deterministic Depth Pro end-to-end CPU reference data."""

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
    parser.add_argument("--size", type=int, default=140)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str((args.repo / "src").resolve()))
    import depth_pro
    from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT

    config = DEFAULT_MONODEPTH_CONFIG_DICT
    config.checkpoint_uri = str(args.checkpoint.resolve())
    model, _ = depth_pro.create_model_and_transforms(
        config=config, device=torch.device("cpu"),
        precision=torch.float32)
    model.eval()
    generator = torch.Generator().manual_seed(20260730)
    rgb = torch.rand(
        3, args.size, args.size, generator=generator)
    normalized = (rgb - 0.5) / 0.5
    with torch.inference_mode():
        output = model.infer(normalized)
    depth = output["depth"].cpu()
    focal = output["focallength_px"].cpu()
    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    rgb.numpy().astype(np.float32).tofile(
        prefix.with_suffix(".rgb.bin"))
    depth.numpy().astype(np.float32).tofile(
        prefix.with_suffix(".depth.bin"))
    np.asarray(focal, dtype=np.float32).tofile(
        prefix.with_suffix(".focal.bin"))
    print(json.dumps({
        "rgb_shape": list(rgb.shape),
        "depth_shape": list(depth.shape),
        "depth_minimum": float(depth.min()),
        "depth_maximum": float(depth.max()),
        "depth_mean": float(depth.mean()),
        "depth_sum": float(depth.double().sum()),
        "focal_length_px": float(focal),
    }, indent=2))


if __name__ == "__main__":
    main()
