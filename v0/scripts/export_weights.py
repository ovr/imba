#!/usr/bin/env python3
"""
Export TemporalAA model weights from Burn's .mpk checkpoint to raw float32 binary.

Usage:
    python export_weights.py <checkpoint.mpk> [output.bin]

The .mpk file is MessagePack-serialized by Burn's CompactRecorder (HalfPrecisionSettings).
Tensors are stored as dicts: {id, param: {bytes, shape, dtype}}, where dtype='F16'.

Output: raw float32 binary (57,352 values = 229,408 bytes) matching AAWeightOffsets.h layout.
"""

import sys
import os

try:
    import msgpack
except ImportError:
    os.system(f"{sys.executable} -m pip install msgpack")
    import msgpack

try:
    import numpy as np
except ImportError:
    os.system(f"{sys.executable} -m pip install numpy")
    import numpy as np


EXPECTED_PARAMS = 57352


def is_param(obj):
    """Check if obj is a Burn ParamSerde: dict with 'id' and 'param' keys."""
    if not isinstance(obj, dict):
        return False
    return b'id' in obj and b'param' in obj and isinstance(obj[b'param'], dict)


def extract_tensor(param_dict):
    """Extract tensor from ParamSerde dict: {id, param: {bytes, shape, dtype}}."""
    p = param_dict[b'param']
    raw_bytes = p[b'bytes']
    shape = p[b'shape']
    dtype_str = p.get(b'dtype', b'F16')

    if dtype_str == b'F16':
        arr = np.frombuffer(raw_bytes, dtype=np.float16).astype(np.float32)
    elif dtype_str == b'F32':
        arr = np.frombuffer(raw_bytes, dtype=np.float32)
    else:
        raise ValueError(f"Unknown dtype: {dtype_str}")

    expected = 1
    for s in shape:
        expected *= s
    assert arr.size == expected, f"Shape mismatch: {arr.size} vs {expected} for shape {shape}"
    return arr, shape


# Config fields to skip during recursive walk (not tensor data)
_SKIP_KEYS = frozenset((
    'stride', 'kernel_size', 'dilation', 'groups', 'padding',
    'num_groups', 'num_channels', 'epsilon', 'affine',
))


def walk_module(obj, tensors, path=""):
    """Recursively walk the module record and extract weight tensors in field order."""
    if obj is None:
        return

    if isinstance(obj, dict):
        if is_param(obj):
            arr, shape = extract_tensor(obj)
            tensors.append((path, arr, shape))
            return

        for key, value in obj.items():
            key_str = key.decode() if isinstance(key, bytes) else str(key)
            if key_str in _SKIP_KEYS:
                continue
            child_path = f"{path}.{key_str}" if path else key_str
            walk_module(value, tensors, child_path)

    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            walk_module(item, tensors, f"{path}[{i}]")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <checkpoint.mpk> [output.bin]")
        sys.exit(1)

    mpk_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) >= 3 else os.path.splitext(mpk_path)[0] + ".bin"

    print(f"Reading {mpk_path}...")
    with open(mpk_path, "rb") as f:
        data = msgpack.unpackb(f.read(), raw=True, strict_map_key=False)

    item = data[b'item']

    print("Extracting tensors...")
    tensors = []
    walk_module(item, tensors)

    print(f"\nFound {len(tensors)} tensors:")
    total_params = 0
    for i, (path, arr, shape) in enumerate(tensors):
        print(f"  [{i:2d}] offset={total_params:6d} count={arr.size:5d}  {path}  shape={list(shape)}")
        total_params += arr.size

    print(f"\nTotal parameters: {total_params}")

    if total_params != EXPECTED_PARAMS:
        print(f"WARNING: expected {EXPECTED_PARAMS} params, got {total_params}")

    all_weights = np.concatenate([arr for _, arr, _ in tensors])
    all_weights.tofile(out_path)

    file_size = os.path.getsize(out_path)
    print(f"Wrote {out_path}: {file_size} bytes ({total_params} float32 values)")


if __name__ == "__main__":
    main()
