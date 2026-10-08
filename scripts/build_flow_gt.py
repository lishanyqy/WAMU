#!/usr/bin/env python
"""Dense 3D scene-flow / 2D point-track targets from the `_extra` GT (entity poses + raw actor ids).

For every episode and probe step k -> k+1 (aligned frames 4k -> 4k+4, 16 raw frames), the dataset's own
`dense_flow` tracks every pixel of frame 4k to frame 4k+4 through its entity's pose change. We sample
query pixels per step, stratified by what the pixel belongs to:
    group 0  robot link            (entities with is_robot)
    group 1  moving object         (non-robot entity, |Δ| > MOVING_MM)
    group 2  static scene          (non-robot entity, |Δ| <= MOVING_MM: table, resting objects)
and store, per query: pixel (u, v) at k, 3D displacement in the head-camera frame (mm), 2D endpoint (u2, v2)
at k+1, and `visible` (in view, unoccluded, same entity) from dense_flow.
Output: artifacts/downstream/gt/<emb>_flow.npz
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, '/home/a1905721/dataspace/openwam-robotwin-gt/tools')
from downstream_data import PROBE_ALIGNED_FRAMES, gt_hdf5
from robotwin_gt_frames import feature_indices
from robotwin_gt_extra import dense_flow, load_extra

GT_DIR = Path('artifacts/downstream/gt')
CAM = 'observation/head_camera'
PER_GROUP = 16      # queries per group per step
MOVING_MM = 1.0


def episode_samples(args):
    emb, key, ep_index, seed = args
    main = gt_hdf5(emb, key)
    extra_path = main.with_name(main.name.replace('.hdf5', '_extra.hdf5'))
    ex = load_extra(extra_path)
    with h5py.File(main, 'r') as f:
        T = len(f[f'{CAM}/rgb'])
        E = f[f'{CAM}/extrinsic_cv'][0].astype(np.float64)           # world -> camera (3x4), static camera
        R, tvec = E[:, :3], E[:, 3]
    raw = feature_indices(T)[PROBE_ALIGNED_FRAMES]
    robot_of = dict(zip(ex['entity_id'].tolist(), ex['is_robot'].tolist()))
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(8):
        t, t2 = int(raw[k]), int(raw[k + 1])
        fl = dense_flow(main, ex, t, t2)
        d_world = (fl['p2'] - fl['p1']) * 1000.0                         # (H, W, 3) mm
        known = np.isfinite(d_world[..., 0])
        is_robot = np.vectorize(lambda e: robot_of.get(int(e), False))(fl['entity_id']) & known
        dist = np.linalg.norm(np.nan_to_num(d_world), axis=-1)
        groups = [is_robot, known & ~is_robot & (dist > MOVING_MM), known & ~is_robot & (dist <= MOVING_MM)]
        for g, mask in enumerate(groups):
            vs, us = np.nonzero(mask)
            if len(vs) == 0:
                continue
            pick = rng.choice(len(vs), size=min(PER_GROUP, len(vs)), replace=False)
            for v, u in zip(vs[pick], us[pick]):
                dc = R @ d_world[v, u]
                p1c = (R @ fl['p1'][v, u] + tvec) * 1000.0                 # start point, camera frame, mm
                rows.append((ep_index, k, g, u, v, *dc, fl['u2'][v, u], fl['v2'][v, u], float(fl['visible'][v, u]), int(fl['entity_id'][v, u]), *p1c))
    return rows


def main():
    for emb in ('agilex', 'franka'):
        keys = list(np.load(GT_DIR / f'{emb}.npz')['keys'])
        jobs = [(emb, k, i, 1000 * i + 7) for i, k in enumerate(keys)]
        with ProcessPoolExecutor(24) as ex:
            rows = [r for rs in ex.map(episode_samples, jobs, chunksize=8) for r in rs]
        arr = np.array(rows, dtype=np.float64)
        out = dict(keys=np.array(keys), episode=arr[:, 0].astype(np.int32), step=arr[:, 1].astype(np.int8), group=arr[:, 2].astype(np.int8),
                   uv=arr[:, 3:5].astype(np.float32), dxyz=arr[:, 5:8].astype(np.float32), uv2=arr[:, 8:10].astype(np.float32),
                   visible=arr[:, 10].astype(bool), entity=arr[:, 11].astype(np.int32), p1=arr[:, 12:15].astype(np.float32),
                   probe_aligned_frames=PROBE_ALIGNED_FRAMES)
        np.savez(GT_DIR / f'{emb}_flow.npz', **out)
        d = np.linalg.norm(out['dxyz'], axis=1)
        print(f'{emb}: {len(arr)} queries over {len(keys)} episodes; per group (robot/moving/static):',
              [int((out["group"] == g).sum()) for g in range(3)], '| median |Δ| mm:', [round(float(np.median(d[out['group'] == g])), 1) for g in range(3)],
              '| visible frac:', [round(float(out['visible'][out['group'] == g].mean()), 3) for g in range(3)])


if __name__ == '__main__':
    main()
