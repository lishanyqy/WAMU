#!/usr/bin/env python
"""Depth Anything 3 (DA3-BASE, monocular: one frame per call) on the GT head-camera frames: backbone tokens for the
dense probes plus a zero-shot relative-depth ceiling. The 8 frames of an episode go through DA3 as a batch of 8
single-view items (api.forward on a (8, 1, 3, H, W) tensor), so no cross-view attention is involved.

Scored frames: aligned frames 4, 8, ..., 32. Frames are processed at 336 x 448 (process_res 448), giving a
24 x 32 token grid that is average-pooled 2 x 2 to 12 x 16 (OpenWAM 8 x 10, DINOv2 16 x 21).
  features    : backbone blocks 3, 5, 7, 9, 11 of the 12-block ViT-B (5, 7, 9, 11 feed DA3's DPT head)  -> (5, 8, 12, 16, 768) fp16
  zs_absrel / zs_delta1 : DA3's predicted depth (relative) resized to the 120 x 160 GT and aligned to the GT by one
                median scale factor per frame (DA3 outputs scale-free depth), then the probe's AbsRel / delta1  -> (8,)
  zs_pairwise : DA-2K-style ordinal accuracy on 200 random pixel pairs per frame whose GT depths differ by >= 10%
                (scale-free, so no alignment)                                                                 -> (8,)
Output: artifacts/<emb>_features_gt/da3/<key>.npz (uncompressed, memory-mappable). Env: depthanything.
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, '/home/a1905721/workspace/Depth-Anything-3/src')
from downstream_data import gt_hdf5
from robotwin_gt_frames import load_row_frames

GT_DIR = Path('artifacts/downstream/gt')
LAYERS = (3, 5, 7, 9, 11)
PROCESS_RES = 448
PROC_H, PROC_W = 336, 448                       # 240 x 320 frames, long side 448 (DA3 upper_bound_resize), multiples of 14
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
PAIR_MIN_RATIO, PAIRS_PER_FRAME = 1.1, 200


def zero_shot(pred, gt, rng):
    """pred: predicted relative depth at GT resolution; gt: mm (0 / non-finite = invalid)."""
    g = gt.astype(np.float32); v = np.isfinite(g) & (g > 0) & np.isfinite(pred) & (pred > 0)
    if v.sum() < 2:
        return np.nan, np.nan, np.nan
    s = np.median(g[v] / pred[v]); p = pred * s
    absrel = float(np.mean(np.abs(p[v] - g[v]) / g[v]))
    delta1 = float(np.mean(np.maximum(p[v] / g[v], g[v] / p[v]) < 1.25))
    gf, pf, vf = g.ravel(), pred.ravel(), v.ravel()
    a, b = rng.integers(0, gf.size, (2, PAIRS_PER_FRAME * 4))
    ok = vf[a] & vf[b] & (np.maximum(gf[a] / np.maximum(gf[b], 1e-6), gf[b] / np.maximum(gf[a], 1e-6)) >= PAIR_MIN_RATIO)
    a, b = a[ok][:PAIRS_PER_FRAME], b[ok][:PAIRS_PER_FRAME]
    pair = float(np.mean(np.sign(pf[a] - pf[b]) == np.sign(gf[a] - gf[b]))) if len(a) else np.nan
    return absrel, delta1, pair


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--embodiment', required=True)
    p.add_argument('--output', type=Path, default=None)
    p.add_argument('--model-dir', default='/home/a1905721/modelspace/depthanything')
    p.add_argument('--device', default='cuda')
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--depth-only', action='store_true', help='store only the predicted depth resized to the GT grid (8, 120, 160) fp16, for offline zero-shot alignment')
    a = p.parse_args()
    out_dir = a.output or Path(f'artifacts/{a.embodiment}_features_gt/' + ('da3_depth' if a.depth_only else 'da3'))
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.disable(logging.INFO)
    from depth_anything_3.api import DepthAnything3
    model = DepthAnything3.from_pretrained(a.model_dir).to(a.device).eval()
    gt = np.load(GT_DIR / f'{a.embodiment}.npz')
    keys = [str(k) for k in gt['keys']][:a.limit]
    depth = gt['depth']                                              # (N, 9, 120, 160) mm, index 1..8 = frames 4..32
    t0 = time.time()
    for i, k in enumerate(keys):
        out = out_dir / f'{k}.npz'
        if out.exists():
            continue
        rng = np.random.default_rng(a.seed + i)
        frames = load_row_frames({'gt_hdf5': str(gt_hdf5(a.embodiment, k))})[1]
        feats = np.empty((len(LAYERS), 8, 12, 16, 768), np.float16)
        zs = np.full((3, 8), np.nan, np.float32)
        x = torch.stack([torch.as_tensor(cv2.resize(np.asarray(frames[4 * (j + 1)]), (PROC_W, PROC_H), interpolation=cv2.INTER_AREA)) for j in range(8)])
        x = ((x.permute(0, 3, 1, 2).float() / 255 - MEAN) / STD).to(a.device)[:, None]          # (8, 1, 3, H, W): 8 single-view items
        with torch.inference_mode():
            o = model.forward(x, export_feat_layers=[] if a.depth_only else list(LAYERS))           # api.forward: autocast bf16
            dep = o['depth'].float().reshape(8, PROC_H, PROC_W).cpu().numpy()
            for li, l in (() if a.depth_only else enumerate(LAYERS)):
                f = o['aux'][f'feat_layer_{l}'].float().reshape(8, PROC_H // 14, PROC_W // 14, -1)  # (8, 24, 32, 768)
                feats[li] = F.avg_pool2d(f.permute(0, 3, 1, 2), 2).permute(0, 2, 3, 1).cpu().numpy()
        pred = np.empty((8,) + depth.shape[2:], np.float32)
        for j in range(8):
            d = np.nan_to_num(dep[j], nan=0, posinf=0, neginf=0)
            pred[j] = d = cv2.resize(d, (depth.shape[3], depth.shape[2]), interpolation=cv2.INTER_AREA)
            zs[:, j] = zero_shot(d, depth[i, j + 1], rng)
        tmp = out.with_suffix('.tmp.npz')
        if a.depth_only:
            np.savez(tmp, pred=pred.astype(np.float16), zs_absrel=zs[0], zs_delta1=zs[1], zs_pairwise=zs[2], key=k, process_res=PROCESS_RES)
            tmp.rename(out)
            continue
        np.savez(tmp, features=feats, zs_absrel=zs[0], zs_delta1=zs[1], zs_pairwise=zs[2], key=k, layers=np.array(LAYERS), process_res=PROCESS_RES)
        tmp.rename(out)
        if (i + 1) % 10 == 0 or i + 1 == len(keys):
            print(f'progress={i + 1}/{len(keys)} ({(time.time() - t0) / (i + 1):.1f} s/ep)', flush=True)
    print(f'complete={len(keys)}', flush=True)


if __name__ == '__main__':
    main()
