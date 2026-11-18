"""Structural checks applied to every submission (also run by submit.py).

A valid `model.safetensors`:
  * is a plain safetensors file (no pickle), parsed here byte by byte,
  * has exactly the reference tensor names, shapes and dtype (F32),
  * has no extra bytes (tensors are contiguous and fill the file exactly),
  * contains only finite values.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import List

from .model import reference_shapes, validate_state_dict

MAX_FILE_SIZE = 16 * 1024 * 1024


def check_safetensors_bytes(data: bytes) -> List[str]:
    errors = []
    if len(data) > MAX_FILE_SIZE:
        return [f"file too large ({len(data)} bytes)"]
    if len(data) < 8:
        return ["file too small"]
    (n,) = struct.unpack("<Q", data[:8])
    if n > len(data) - 8:
        return ["corrupted header length"]
    try:
        header = json.loads(data[8:8 + n])
    except Exception as e:  # noqa: BLE001
        return [f"header is not JSON: {e}"]
    header.pop("__metadata__", None)
    ref = reference_shapes()
    if set(header) != set(ref):
        errors.append(f"tensor names differ: missing={sorted(set(ref) - set(header))[:5]} "
                      f"extra={sorted(set(header) - set(ref))[:5]}")
        return errors
    spans = []
    for k, info in header.items():
        if info.get("dtype") != "F32":
            errors.append(f"{k}: dtype {info.get('dtype')} != F32")
        if tuple(info.get("shape", ())) != ref[k]:
            errors.append(f"{k}: shape {info.get('shape')} != {list(ref[k])}")
        a, b = info["data_offsets"]
        numel = 1
        for d in ref[k]:
            numel *= d
        if b - a != 4 * numel:
            errors.append(f"{k}: byte span {b - a} != {4 * numel}")
        spans.append((a, b))
    spans.sort()
    pos = 0
    for a, b in spans:
        if a != pos:
            errors.append("tensor data is not contiguous (gap or overlap)")
            break
        pos = b
    if 8 + n + pos != len(data):
        errors.append(f"file has {len(data) - 8 - n - pos} unexpected trailing bytes")
    if errors:
        return errors[:5] + ([f"... and {len(errors) - 5} more"] if len(errors) > 5 else [])
    from safetensors.torch import load as st_load

    err = validate_state_dict(st_load(data))
    return [err] if err else []


def check_safetensors_file(path: str | Path) -> List[str]:
    return check_safetensors_bytes(Path(path).read_bytes())
