#!/usr/bin/env python3
"""Validate replayed RoboTwin GT RGB against the original RGB HDF5."""
import argparse
from pathlib import Path
import cv2
import h5py
import numpy as np


def decode(x):
    return cv2.imdecode(np.frombuffer(x, np.uint8), cv2.IMREAD_COLOR)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--original", type=Path, required=True)
    p.add_argument("--ground-truth", type=Path, required=True)
    p.add_argument("--camera", default="head_camera")
    p.add_argument("--samples", type=int, default=8)
    args = p.parse_args()
    rows = []
    for src in sorted(args.original.glob("episode*.hdf5")):
        dst = args.ground_truth / src.name
        if not dst.exists():
            continue
        with h5py.File(src, "r") as a, h5py.File(dst, "r") as b:
            x = a[f"observation/{args.camera}/rgb"]
            y = b[f"observation/{args.camera}/rgb"]
            if len(x) != len(y):
                rows.append((src.stem, len(x), len(y), np.inf))
                continue
            ids = np.linspace(0, len(x) - 1, min(args.samples, len(x)), dtype=int)
            errors = []
            for i in ids:
                u, v = decode(x[i]), decode(y[i])
                errors.append(float(np.mean(np.abs(u.astype(np.float32) - v.astype(np.float32)))))
            rows.append((src.stem, len(x), len(y), float(np.mean(errors))))
    if not rows:
        raise SystemExit("no comparable episodes")
    for row in rows:
        print("episode=%s original=%d replay=%d mean_abs=%.4f" % row)
    bad = [r for r in rows if not np.isfinite(r[3]) or r[3] > 5.0]
    print(f"checked={len(rows)} bad={len(bad)}")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
