#!/usr/bin/env python3
"""Compute one shared normalization stats.json for all WAMU-real buckets.

Pools every frame of ``action`` and ``observation.state`` across the
bucket subdirs (``bowls``, ``cupstack``, ...) of a ``WAMsU/wamu_real`` checkout and writes
``<root>/meta/stats.json`` in the same schema as the per-bucket
``meta/stats.json`` (min/max/mean/std/count/q01/q10/q50/q90/q99, 14-D),
plus ``<root>/meta/normalization_stats.npy`` -- the deploy denormalizer
artifact ``{"joint": {mean, std, min, max, q01, q99}}`` in raw 14-D space that
the trainer copies into each checkpoint dir.

The 7-D single-arm bucket (headphones) uses the left-arm slots, so it only
contributes to dims 0-6; ``count`` is therefore per dimension.

    python scripts/compute_wamu_real_stats.py [--root datasets/lishanyang_wamu_real]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

FULL_DIM = 14
QUANTILES = {"q01": 0.01, "q10": 0.10, "q50": 0.50, "q90": 0.90, "q99": 0.99}
COLUMNS = ("action", "observation.state")
DEPLOY_KEYS = ("mean", "std", "min", "max", "q01", "q99")


def load_bucket(bucket: Path, column: str) -> np.ndarray:
    files = sorted((bucket / "data").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no data parquet files under {bucket}/data")
    parts = [np.stack(pq.read_table(f, columns=[column]).column(column).to_pylist()) for f in files]
    values = np.concatenate(parts).astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{bucket.name}/{column} has non-finite values")
    return values


def pooled_stats(per_dim: list[np.ndarray]) -> dict:
    out = {"min": [], "max": [], "mean": [], "std": [], "count": []}
    out.update({k: [] for k in QUANTILES})
    for values in per_dim:
        out["min"].append(float(values.min()))
        out["max"].append(float(values.max()))
        out["mean"].append(float(values.mean()))
        out["std"].append(float(values.std()))
        out["count"].append(int(values.size))
        for key, q in QUANTILES.items():
            out[key].append(float(np.quantile(values, q)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="datasets/lishanyang_wamu_real")
    args = parser.parse_args()
    root = Path(args.root)
    buckets = [p for p in sorted(root.iterdir()) if p.is_dir() and (p / "meta" / "info.json").is_file()]
    if not buckets:
        raise FileNotFoundError(f"no <bucket>/meta/info.json under {root}")

    result: dict = {}
    for column in COLUMNS:
        per_dim: list[list[np.ndarray]] = [[] for _ in range(FULL_DIM)]
        for bucket in buckets:
            values = load_bucket(bucket, column)
            if values.shape[1] not in (7, FULL_DIM):
                raise ValueError(f"{bucket.name}/{column}: unexpected width {values.shape[1]}")
            for dim in range(values.shape[1]):  # 7-D bucket fills left-arm dims 0-6
                per_dim[dim].append(values[:, dim])
            print(f"{bucket.name:22s} {column:18s} frames={values.shape[0]} dims={values.shape[1]}")
        result[column] = pooled_stats([np.concatenate(v) for v in per_dim])

    result["_meta"] = {
        "buckets": [b.name for b in buckets],
        "note": "Pooled over all buckets; 7-D buckets contribute to dims 0-6 (left arm) only.",
    }
    out = root / "meta" / "stats.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    deploy = {"joint": {k: np.asarray(result["action"][k], dtype=np.float32) for k in DEPLOY_KEYS}}
    npy = root / "meta" / "normalization_stats.npy"
    np.save(npy, deploy, allow_pickle=True)
    print(f"wrote {npy}")
    for key in ("min", "max", "mean", "std", "count"):
        print(f"  action {key:5s}", np.round(result["action"][key], 3).tolist())


if __name__ == "__main__":
    main()
