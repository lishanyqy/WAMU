#!/usr/bin/env python
"""Build a deterministic Robotwin 2.0 video manifest for OpenWAM experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import av


def video_info(path: Path) -> tuple[int, float, int, int]:
    with av.open(str(path), options={"hwaccel": "none"}) as container:
        stream = container.streams.video[0]
        frames = int(stream.frames or 0)
        fps = float(stream.average_rate) if stream.average_rate else 0.0
        duration = float(stream.duration * stream.time_base) if stream.duration else 0.0
        width = int(stream.width or 0)
        height = int(stream.height or 0)
        if duration <= 0 and frames > 0 and fps > 0:
            duration = frames / fps
        return frames, duration, width, height


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clip-seconds", type=float, default=10.0)
    parser.add_argument("--num-frames", type=int, default=33)
    parser.add_argument("--frame-stride", type=int, default=4)
    args = parser.parse_args()

    rows = []
    for path in sorted(args.root.rglob("*.mp4")):
        task = path.parents[2].name
        variant = path.parents[1].name
        episode = path.stem
        row = {
            "task": task,
            "variant": variant,
            "episode": episode,
            "video": str(path.resolve()),
            "clip_start": 0.0,
            "clip_end": args.clip_seconds,
            "clip_seconds_requested": args.clip_seconds,
            "num_frames": args.num_frames,
            "frame_stride": args.frame_stride,
            "sampling_policy": "uniform_within_clip_with_stride_cap",
        }
        try:
            frames, duration, width, height = video_info(path)
            row.update({"frames": frames, "duration": duration, "width": width, "height": height})
            row["clip_end"] = min(args.clip_seconds, duration) if duration > 0 else args.clip_seconds
            row["valid"] = bool(frames > 0 and duration > 0 and width > 0 and height > 0)
            row["error"] = None
        except Exception as exc:  # keep a complete audit record
            row.update({"frames": 0, "duration": 0.0, "width": 0, "height": 0, "valid": False, "error": repr(exc)})
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(json.dumps({"rows": len(rows), "valid": sum(r["valid"] for r in rows), "sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
