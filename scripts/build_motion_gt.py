#!/usr/bin/env python
"""Sparse two-frame 3D motion targets for the D4RT-style probe (query pixel at probe frame k -> 3D displacement to k+1).

Tracked points per probe step k -> k+1 (aligned frames 4k -> 4k+4, i.e. 16 raw frames):
  kind 0 / 1  left / right end-effector: exact. `endpose/*_endpose[:, :3]` is the world-frame TCP position;
              projected with extrinsic_cv / intrinsic_cv. Kept when the TCP pixel is inside the image at k.
  kind 2      manipulated object: approximate. Centroid of the target mask (artifacts/downstream/gt, 120 x 160)
              back-projected through the GT depth; kept when the mask has >= MIN_PX pixels at both k and k+1.
All 3D quantities are in the head-camera frame in mm (the camera is static, so camera-frame displacement equals
world-frame displacement up to rotation). Output: artifacts/downstream/gt/<emb>_motion.npz with one row per sample.
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from downstream_data import PROBE_ALIGNED_FRAMES, gt_hdf5
from robotwin_gt_frames import feature_indices

GT_DIR = Path('artifacts/downstream/gt')
CAM = 'observation/head_camera'
W, H = 320, 240
MIN_PX = 20


def episode_samples(args):
    emb, key, ep_index, target, depth = args
    with h5py.File(gt_hdf5(emb, key), 'r') as f:
        T = len(f[f'{CAM}/rgb'])
        K = f[f'{CAM}/intrinsic_cv'][0].astype(np.float64)
        E = f[f'{CAM}/extrinsic_cv'][0].astype(np.float64)       # world -> camera (OpenCV), static
        arms = [f['endpose/left_endpose'][:, :3], f['endpose/right_endpose'][:, :3]]
    raw = feature_indices(T)[PROBE_ALIGNED_FRAMES]                 # raw frame index of each probe frame
    rows = []

    def cam_mm(xw):
        return (E @ np.append(xw, 1.0)) * 1000.0

    def pix(xc):
        uv = K @ (xc / 1000.0)
        return uv[:2] / uv[2]

    for k in range(8):
        t, t2 = raw[k], raw[k + 1]
        for kind, P in enumerate(arms):
            if np.allclose(P[:, :3], 0):
                continue
            a, b = cam_mm(P[t]), cam_mm(P[t2])
            uv = pix(a)
            if not (0 <= uv[0] < W and 0 <= uv[1] < H and a[2] > 0):
                continue
            uv2 = pix(b)
            rows.append((ep_index, k, kind, *uv, *a, *b, *uv2, float(0 <= uv2[0] < W and 0 <= uv2[1] < H)))
        # manipulated object: visible-surface centroid at the probe resolution (120 x 160)
        sK = K.copy(); sK[0] *= 160 / W; sK[1] *= 120 / H
        cents = []
        for kk in (k, k + 1):
            ys, xs = np.nonzero(target[kk])
            z = depth[kk, ys, xs].astype(np.float64)
            ok = np.isfinite(z) & (z > 0)
            if ok.sum() < MIN_PX:
                cents.append(None); continue
            z, xs, ys = z[ok], xs[ok], ys[ok]
            X = (xs + 0.5 - sK[0, 2]) / sK[0, 0] * z
            Y = (ys + 0.5 - sK[1, 2]) / sK[1, 1] * z
            cents.append(np.array([X.mean(), Y.mean(), z.mean()]))
        if cents[0] is not None and cents[1] is not None:
            a, b = cents
            rows.append((ep_index, k, 2, *pix(a), *a, *b, *pix(b), 1.0))
    return rows


def main():
    for emb in ('agilex', 'franka'):
        gt = np.load(GT_DIR / f'{emb}.npz')
        keys = list(gt['keys'])
        jobs = [(emb, k, i, gt['target'][i], gt['depth'][i]) for i, k in enumerate(keys)]
        with ThreadPoolExecutor(16) as ex:   # I/O bound (HDF5 reads); threads avoid pickling the mask/depth arrays
            rows = [r for rs in ex.map(episode_samples, jobs) for r in rs]
        arr = np.array(rows, dtype=np.float64)
        out = dict(keys=np.array(keys), episode=arr[:, 0].astype(np.int32), step=arr[:, 1].astype(np.int8), kind=arr[:, 2].astype(np.int8),
                   uv=arr[:, 3:5].astype(np.float32), xyz=arr[:, 5:8].astype(np.float32), xyz2=arr[:, 8:11].astype(np.float32),
                   uv2=arr[:, 11:13].astype(np.float32), target_in_view=arr[:, 13].astype(bool), probe_aligned_frames=PROBE_ALIGNED_FRAMES)
        np.savez(GT_DIR / f'{emb}_motion.npz', **out)
        d = np.linalg.norm(out['xyz2'] - out['xyz'], axis=1)
        print(f'{emb}: {len(arr)} samples over {len(keys)} episodes; by kind (left/right/object):',
              [int((out['kind'] == c).sum()) for c in range(3)],
              '| median |Δ| mm by kind:', [round(float(np.median(d[out["kind"] == c])), 1) for c in range(3) if (out['kind'] == c).any()])


if __name__ == '__main__':
    main()
