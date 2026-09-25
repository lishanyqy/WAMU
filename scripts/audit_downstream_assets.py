#!/usr/bin/env python3
"""Audit Robotwin supervision and extracted feature granularity for downstream probes."""
import argparse, glob, json, os
from collections import Counter

import h5py
import numpy as np


def h5_info(path):
    out = {"path": path, "datasets": {}, "frames": None}
    with h5py.File(path, "r") as f:
        for name, obj in f.items():
            pass
        def walk(group, prefix=""):
            for name, obj in group.items():
                key = f"{prefix}/{name}" if prefix else name
                if isinstance(obj, h5py.Dataset):
                    out["datasets"][key] = {
                        "shape": list(obj.shape), "dtype": str(obj.dtype)
                    }
                    if out["frames"] is None and obj.ndim and obj.shape[0] > 1:
                        out["frames"] = int(obj.shape[0])
                else:
                    walk(obj, key)
        walk(f)
    return out


def feature_info(root):
    out = {"root": root, "files": 0, "shapes": Counter(), "spatial_or_temporal": 0}
    for path in glob.glob(os.path.join(root, "*.npz")):
        try:
            z = np.load(path, allow_pickle=False)
            shape = tuple(z["features"].shape)
            out["files"] += 1
            out["shapes"][str(shape)] += 1
            # Existing downstream-compatible features need sample/frame/token axes.
            if z["features"].ndim >= 3:
                out["spatial_or_temporal"] += 1
        except Exception as exc:
            out.setdefault("errors", []).append({"path": path, "error": str(exc)})
    out["shapes"] = dict(out["shapes"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="/home/a1905721/dataspace/openwam/benchmark_data/robotwin2.0/dataset")
    ap.add_argument("--feature-roots", nargs="*", default=[])
    ap.add_argument("--output", default="artifacts/downstream_asset_audit.json")
    args = ap.parse_args()

    h5_paths = sorted(glob.glob(os.path.join(args.dataset, "*", "*", "data", "episode*.hdf5")))
    sample_paths = h5_paths[:3]
    samples = [h5_info(p) for p in sample_paths]
    dataset_counts = Counter()
    for sample in samples:
        for key in sample["datasets"]:
            dataset_counts[key.split("/")[0] if "/" in key else key] += 1
    result = {
        "dataset": args.dataset,
        "hdf5_count": len(h5_paths),
        "sample_count": len(samples),
        "sample_hdf5": samples,
        "top_level_modalities_in_samples": dict(dataset_counts),
        "supervision_status": {
            "rgb": any("/rgb" in k for s in samples for k in s["datasets"]),
            "depth": any("depth" in k.lower() for s in samples for k in s["datasets"]),
            "mask_or_segmentation": any(any(x in k.lower() for x in ("mask", "seg", "label")) for s in samples for k in s["datasets"]),
            "camera_calibration": any("intrinsic" in k or "extrinsic" in k for s in samples for k in s["datasets"]),
            "action_or_pose": any(any(x in k for x in ("joint_action", "endpose")) for s in samples for k in s["datasets"]),
            "pointcloud_nonempty": any(s["datasets"].get("pointcloud", {}).get("shape", [0, 0])[-1] > 0 for s in samples),
        },
        "features": [feature_info(r) for r in args.feature_roots],
        "interpretation": {
            "dense_tasks": "Require per-frame/spatial tokens and pixel/point labels; pooled (layers,dim) features are insufficient.",
            "available_now": "RGB/action/pose-aligned pilot probes and layer-selection infrastructure.",
        },
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
