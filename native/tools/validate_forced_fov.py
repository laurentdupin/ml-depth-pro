"""Validate InferBridge BGRA and forced-FOV behavior against the proven ABI."""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dll", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--reference-file", type=Path)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--fov", type=float, default=63.0)
    args = parser.parse_args()

    rng = np.random.default_rng(20260730)
    bgra = rng.integers(
        0, 256, (args.height, args.width, 4), dtype=np.uint8)
    bgra[:, :, 3] = 255
    rgb = np.stack(
        [bgra[:, :, channel] for channel in range(3)]).astype(
            np.float32) / 255.0

    forced_focal_value = (
        0.5 * args.width /
        np.tan(np.deg2rad(args.fov) * 0.5))
    if args.reference_file and args.reference_file.exists():
        reference = np.load(args.reference_file)
    else:
        torchvision_library = torch.library.Library("torchvision", "DEF")
        torchvision_library.define(
            "nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")
        sys.path.insert(0, str((args.repo / "src").resolve()))
        import depth_pro
        from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT

        config = DEFAULT_MONODEPTH_CONFIG_DICT
        config.checkpoint_uri = str(args.checkpoint.resolve())
        model, _ = depth_pro.create_model_and_transforms(
            config=config, device=torch.device("cpu"),
            precision=torch.float32)
        model.eval()
        with torch.inference_mode():
            reference = model.infer(
                torch.from_numpy((rgb - 0.5) / 0.5),
                f_px=forced_focal_value)["depth"].numpy()
        if args.reference_file:
            args.reference_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(args.reference_file, reference)

    library = ctypes.CDLL(str(args.dll.resolve()))
    library.depth_pro_create_vulkan.argtypes = [
        ctypes.c_char_p, ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_void_p)]
    library.depth_pro_create_vulkan.restype = ctypes.c_int
    common_tail = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_float)]
    library.depth_pro_infer_rgb_f32.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32, ctypes.c_int32, *common_tail]
    library.depth_pro_infer_rgb_f32.restype = ctypes.c_int
    library.depth_pro_infer_bgra8_f32.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_uint64, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_float, *common_tail]
    library.depth_pro_infer_bgra8_f32.restype = ctypes.c_int
    library.depth_pro_last_error.restype = ctypes.c_char_p
    library.depth_pro_destroy.argtypes = [ctypes.c_void_p]

    context = ctypes.c_void_p()
    status = library.depth_pro_create_vulkan(
        str(args.model.resolve()).encode(), args.device,
        ctypes.byref(context))
    if status:
        raise RuntimeError(library.depth_pro_last_error().decode())
    forced = np.empty((args.height, args.width), np.float32)
    forced_focal = ctypes.c_float()
    try:
        status = library.depth_pro_infer_bgra8_f32(
            context,
            bgra.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            bgra.strides[0], args.width, args.height, args.fov,
            forced.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            forced.size, ctypes.byref(forced_focal))
        if status:
            raise RuntimeError(library.depth_pro_last_error().decode())
    finally:
        library.depth_pro_destroy(context)

    tangent_forced = np.tan(np.deg2rad(args.fov) * 0.5)
    difference = np.abs(forced - reference)
    relative = difference / np.maximum(np.abs(reference), 1e-6)
    reference_worker = (
        (reference - reference.min()) /
        (25.0 - reference.min()))
    actual_worker = (
        (forced - forced.min()) /
        (25.0 - forced.min()))
    worker_difference = np.abs(actual_worker - reference_worker)
    expected_focal = 0.5 * args.width / tangent_forced
    report = {
        "forced_focal": forced_focal.value,
        "expected_forced_focal": expected_focal,
        "maximum_absolute_error": float(difference.max()),
        "mean_absolute_error": float(difference.mean()),
        "maximum_relative_error": float(relative.max()),
        "mean_relative_error": float(relative.mean()),
        "maximum_worker_output_error": float(worker_difference.max()),
        "mean_worker_output_error": float(worker_difference.mean()),
    }
    print(json.dumps(report, indent=2))
    if worker_difference.max() > 0.01 or abs(
            forced_focal.value - expected_focal) > 1e-4:
        raise SystemExit("Depth Pro forced-FOV gate failed")


if __name__ == "__main__":
    main()
