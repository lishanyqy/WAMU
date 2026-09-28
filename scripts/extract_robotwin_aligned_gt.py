#!/usr/bin/env python3
"""Extract GT labels at exactly the frames used by Robotwin temporal features."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def feature_indices(total, n=33, stride=4):
    if total <= 0:
        raise ValueError("empty episode")
    ids = np.arange(n, dtype=np.int64) * stride
    if ids[-1] >= total:
        ids = np.linspace(0, total - 1, n, dtype=np.int64)
    return ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--gt-root", type=Path, required=True,
                   help="dataset_gt/<embodiment>, containing task/gt_export/data")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--camera", default="head_camera")
    p.add_argument("--num-frames", type=int, default=33)
    p.add_argument("--frame-stride", type=int, default=4)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for line in args.manifest.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        task, episode = row["task"], row["episode"]
        src = args.gt_root / task / "gt_export" / "data" / f"{episode}.hdf5"
        out = args.output / f"{task}__{episode}.npz"
        if not src.exists():
            rows.append({"task": task, "episode": episode, "status": "missing", "path": str(src)})
            continue
        with h5py.File(src, "r") as f:
            prefix = f"observation/{args.camera}"
            total = len(f[f"{prefix}/rgb"])
            ids = feature_indices(total, args.num_frames, args.frame_stride)
            data = {"frame_indices": ids, "total_frames": total}
            for name in ("depth", "actor_segmentation", "mesh_segmentation",
                         "intrinsic_cv", "extrinsic_cv", "cam2world_gl"):
                key = f"{prefix}/{name}"
                if key in f:
                    data[name] = f[key][ids]
            if "pointcloud" in f:
                # Pointcloud is stored at the root and is frame-aligned.
                data["pointcloud"] = f["pointcloud"][ids]
        np.savez_compressed(out, **data, task=task, episode=episode,
                            camera=args.camera, num_frames=args.num_frames,
                            frame_stride=args.frame_stride)
        rows.append({"task": task, "episode": episode, "status": "ok",
                     "total_frames": total, "indices": ids.tolist(), "path": str(out)})
    (args.output / "index.json").write_text(json.dumps(rows, indent=2))
    print(f"ok={sum(x['status']=='ok' for x in rows)} missing={sum(x['status']=='missing' for x in rows)}")


if __name__ == "__main__":
    main()
