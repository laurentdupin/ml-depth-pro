"""Convert the official Depth Pro checkpoint to bounded native tensor data."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import zlib
from pathlib import Path
from typing import Any

MAGIC = b"DPROFMOD"
FORMAT_VERSION = 1
ENDIAN_TAG = 0x01020304
DTYPE_FLOAT16 = 2
HEADER = struct.Struct("<8sIIIIQQQQQ")
RECORD = struct.Struct("<112sII4QQQQIIQ")
METADATA = struct.Struct("<8sIIIIII32s64s")
ALIGNMENT = 64
CONVERTER_ID = "depth-pro-export-pytorch-v1"


def align(value: int) -> int:
    return (value + ALIGNMENT - 1) & ~(ALIGNMENT - 1)


def sha256_file(path: Path) -> bytes:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.digest()


def tensor_bytes(tensor: Any) -> bytes:
    import torch

    if tensor.dtype != torch.float16:
        raise TypeError(f"unsupported tensor dtype: {tensor.dtype}")
    return tensor.detach().cpu().contiguous().numpy().tobytes(order="C")


def main() -> None:
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")

    canonical_sha = sha256_file(args.checkpoint)
    state = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise TypeError("checkpoint is not a state dictionary")
    tensors = []
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"state entry is not a tensor: {name}")
        encoded_name = name.encode("utf-8")
        if not encoded_name or len(encoded_name) >= 112:
            raise ValueError(f"invalid tensor name: {name}")
        if value.dtype != torch.float16 or not 1 <= value.ndim <= 4:
            raise TypeError(
                f"unsupported tensor {name}: {value.dtype}/{value.ndim}")
        payload = tensor_bytes(value)
        tensors.append((
            name, tuple(value.shape), value, len(payload),
            zlib.crc32(payload)))

    directory_offset = HEADER.size
    directory_bytes = len(tensors) * RECORD.size
    metadata_offset = directory_offset + directory_bytes
    data_offset = align(metadata_offset + METADATA.size)
    cursor = data_offset
    records = []
    for name, shape, _, payload_bytes, checksum in tensors:
        cursor = align(cursor)
        dimensions = list(shape) + [0] * (4 - len(shape))
        elements = 1
        for dimension in shape:
            elements *= dimension
        if elements * 2 != payload_bytes:
            raise RuntimeError(f"tensor byte mismatch: {name}")
        records.append(RECORD.pack(
            name.encode("utf-8"), DTYPE_FLOAT16, len(shape),
            *dimensions, cursor, payload_bytes, elements,
            checksum, 0, 0))
        cursor += payload_bytes

    header = HEADER.pack(
        MAGIC, FORMAT_VERSION, ENDIAN_TAG, 0, len(tensors),
        directory_offset, directory_bytes, data_offset, cursor,
        metadata_offset)
    metadata = METADATA.pack(
        b"DPROFMTA", 1, METADATA.size, FORMAT_VERSION, 0, 0, 0,
        canonical_sha, CONVERTER_ID.encode("ascii"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as output:
        output.write(header)
        for record in records:
            output.write(record)
        output.write(metadata)
        output.write(b"\0" * (data_offset - output.tell()))
        for tensor, record in zip(tensors, records):
            offset = RECORD.unpack(record)[7]
            output.write(b"\0" * (offset - output.tell()))
            output.write(tensor_bytes(tensor[2]))
        if output.tell() != cursor:
            raise RuntimeError("native file length mismatch")

    cache_key = (
        f"depth-pro:{canonical_sha.hex()}:"
        f"converter={CONVERTER_ID}:format={FORMAT_VERSION}")
    print(json.dumps({
        "format": "DPROFMOD",
        "format_version": FORMAT_VERSION,
        "tensor_count": len(tensors),
        "bytes": cursor,
        "canonical_sha256": canonical_sha.hex(),
        "converter": CONVERTER_ID,
        "cache_key": cache_key,
        "output": str(args.output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()

