#!/usr/bin/env python
from __future__ import annotations

import argparse, json, os
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


def read_frames(path: str, n: int, stride: int) -> list[Image.Image]:
    with av.open(path, options={"hwaccel": "none"}) as c:
        stream = c.streams.video[0]
        total = int(stream.frames or 0)
        indices = np.linspace(0, max(0, total - 1), n, dtype=np.int64) if total <= n * stride else np.arange(n) * stride
        wanted = set(int(x) for x in indices)
        out = {}
        for i, frame in enumerate(c.decode(stream)):
            if i in wanted:
                out[i] = frame.to_image().convert("RGB")
            if i > max(wanted):
                break
        if len(out) < n:
            raise RuntimeError(f"decoded {len(out)}/{n} frames from {path}")
        return [out[int(i)] for i in indices]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default="facebook/dinov2-large")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-frames", type=int, default=33)
    p.add_argument("--frame-stride", type=int, default=4)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA requested but unavailable: torch={torch.__version__}, build={torch.version.cuda}")
        torch.cuda.set_device(args.device)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    print(f"device={device} name={torch.cuda.get_device_name(device)} dtype={dtype} torch={torch.__version__} cuda_build={torch.version.cuda}", flush=True)
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, torch_dtype=dtype, low_cpu_mem_usage=True).to(device).eval()
    print(f"model_device={next(model.parameters()).device} model_dtype={next(model.parameters()).dtype}", flush=True)
    rows = [json.loads(x) for x in args.manifest.read_text().splitlines() if x.strip()]
    done = 0
    for row in rows:
        out = args.output / f"{row['task']}__{row['episode']}.npz"
        if out.exists():
            done += 1
            continue
        frames = read_frames(row["video"], args.num_frames, args.frame_stride)
        feats = []
        with torch.inference_mode():
            for start in range(0, len(frames), args.batch_size):
                inputs = processor(images=frames[start:start + args.batch_size], return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                hidden = model(**inputs, output_hidden_states=True).hidden_states
                feats.append(torch.stack([h[:, 0] for h in hidden]).transpose(0, 1).cpu().to(torch.float16).numpy())
        feat = np.concatenate(feats, axis=0).mean(axis=0)
        np.savez_compressed(out, features=feat.astype(np.float16), task=row["task"], episode=row["episode"], model=args.model, dtype=str(dtype), num_frames=args.num_frames, frame_stride=args.frame_stride)
        done += 1
        if done % 10 == 0:
            print(f"progress={done}/{len(rows)}", flush=True)
    print(f"complete={done}/{len(rows)} dtype={dtype} device={device}")


if __name__ == "__main__":
    main()
