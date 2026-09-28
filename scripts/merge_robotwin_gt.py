#!/usr/bin/env python3
"""Merge RoboTwin per-frame GT pickle files without requiring ffmpeg."""
import argparse
import os
import pickle
from pathlib import Path

from tqdm import tqdm


def load(path):
    with path.open("rb") as f:
        return pickle.load(f)


def parse_structure(value):
    if isinstance(value, dict):
        return {k: parse_structure(v) for k, v in value.items()}
    return []


def append(dst, src):
    for key in dst:
        if key not in src:
            continue
        if isinstance(dst[key], dict):
            append(dst[key], src[key])
        else:
            dst[key].append(src[key])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--episode", type=int, required=True)
    args = p.parse_args()

    folder = args.cache_root / f"episode{args.episode}"
    files = sorted(folder.glob("*.pkl"), key=lambda x: int(x.stem))
    if not files or [int(x.stem) for x in files] != list(range(len(files))):
        raise RuntimeError(f"incomplete frame cache: {folder}")

    data = parse_structure(load(files[0]))
    for path in tqdm(files, desc=f"episode{args.episode}"):
        append(data, load(path))

    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root / f"episode{args.episode}.hdf5"
    import cv2
    import h5py

    def create_hdf5_from_dict(group, value):
        for key, item in value.items():
            if isinstance(item, dict):
                create_hdf5_from_dict(group.create_group(key), item)
            elif isinstance(item, list):
                array = __import__("numpy").array(item)
                if "rgb" in key:
                    encoded = []
                    for image in array:
                        ok, data = cv2.imencode(".jpg", image)
                        if not ok:
                            raise RuntimeError(f"JPEG encoding failed for {key}")
                        encoded.append(data.tobytes())
                    width = max(map(len, encoded))
                    group.create_dataset(key, data=encoded, dtype=f"S{width}")
                else:
                    group.create_dataset(key, data=array)
    with h5py.File(output, "w") as f:
        create_hdf5_from_dict(f, data)
    print(f"wrote={output} frames={len(files)}")


if __name__ == "__main__":
    main()
