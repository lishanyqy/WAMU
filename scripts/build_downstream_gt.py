#!/usr/bin/env python
"""Probe targets at the 9 latent-aligned frames, for depth / target-segmentation / task probes.

Target object (GT segmentation stores palette colours with no colour->name table):
  * background/robot colours = actor colours visible at frame 0 in *every* task of the embodiment
    (table, floor, always-visible gripper parts); robot links that enter later are never at frame 0;
  * candidates = the other actor colours visible at frame 0 (the task's objects);
  * target = the candidate whose mask centroid moves furthest over the 33 aligned frames,
    i.e. the object the robot manipulates (e.g. the hammer, not the block; A, not B).
Depth (mm) and the target mask are nearest-downsampled to --size, so no depth is averaged across edges.
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from downstream_data import PROBE_ALIGNED_FRAMES, gt_hdf5, load_split
from robotwin_gt_frames import feature_indices

CAM = 'observation/head_camera'
OFFICIAL = Path.home() / 'dataspace/openwam/benchmark_data/robotwin2.0/dataset'
MIN_AREA = 5e-4  # fraction of the image a frame-0 actor must cover to count as an object


def frame0_colours(path):
    with h5py.File(path, 'r') as f:
        s = f[f'{CAM}/actor_segmentation'][0].reshape(-1, 3)
    u, c = np.unique(s, axis=0, return_counts=True)
    return {tuple(int(v) for v in x) for x, n in zip(u, c) if n / len(s) >= MIN_AREA}


def _palette():
    """RoboTwin's 256-colour segmentation palette (envs/camera/camera.py with the GT patch): colour = palette[id % 256]."""
    from PIL import ImageColor
    pal = [ImageColor.getrgb(c) for c in sorted(set(ImageColor.colormap.values()))]
    used, state = set(pal), 12345
    while len(pal) < 256:
        state = (state * 1103515245 + 12345) % 2 ** 31
        c = ((state >> 16) & 255, (state >> 8) & 255, state & 255)
        if c not in used:
            used.add(c)
            pal.append(c)
    return pal


def clutter_colours(main):
    """Actor colours of the cluttered-table distractors (randomized scenes; empty for clean ones).

    The clean target rule picks among the task's objects; with clutter on the table a static task object (a clicked
    bell) loses to a distractor that the arm partly occludes, so distractors are excluded to keep the same rule."""
    real = Path(main).resolve()
    # The replay rewrites its own scene_info.json and can leave cluttered_table_info empty; read the official record.
    task_dir = real.parent.parent
    info_path = OFFICIAL / task_dir.parent.name / 'aloha-agilex_randomized_500' / 'scene_info.json' \
        if task_dir.name == 'gt_export_randomized' else task_dir / 'scene_info.json'
    extra = real.with_name(real.name.replace('.hdf5', '_extra.hdf5'))
    if not info_path.exists() or not extra.exists():
        return set()
    ep = 'episode_' + real.stem.replace('episode', '')
    types = {c['object_type'] for c in json.loads(info_path.read_text()).get(ep, {}).get('cluttered_table_info', [])}
    if not types:
        return set()
    with h5py.File(extra, 'r') as x:
        ids = [int(e) for e, n in zip(x['entities/entity_id'][:], x['entities/name'][:].astype(str)) if n in types]
    pal = _palette()
    return {tuple(pal[e % 256]) for e in ids}


def episode_targets(args):
    embodiment, key, background, size = args
    clutter = clutter_colours(gt_hdf5(embodiment, key))
    with h5py.File(gt_hdf5(embodiment, key), 'r') as f:
        n = len(f[f'{CAM}/rgb'])
        ids = feature_indices(n)
        seg = f[f'{CAM}/actor_segmentation'][ids]  # (33, H, W, 3)
        depth = f[f'{CAM}/depth'][ids[PROBE_ALIGNED_FRAMES]].astype(np.float32)  # (9, H, W) mm
    h, w = seg.shape[1:3]
    ys, xs = np.mgrid[:h, :w]
    u, c = np.unique(seg[0].reshape(-1, 3), axis=0, return_counts=True)
    candidates = [tuple(int(v) for v in x) for x, k in zip(u, c) if k / (h * w) >= MIN_AREA]
    candidates = [col for col in candidates if col not in background and col not in clutter]
    moves = {}
    for col in candidates:
        m = (seg == np.array(col, np.uint8)).all(-1)
        cen = np.array([(ys[mm].mean(), xs[mm].mean()) if mm.any() else (np.nan, np.nan) for mm in m])
        moves[col] = float(np.nanmax(np.linalg.norm(cen - cen[0], axis=1)))
    target = max(moves, key=moves.get) if moves else None
    if target is None:
        mask = np.zeros((len(PROBE_ALIGNED_FRAMES), h, w), bool)
    else:
        mask = (seg[PROBE_ALIGNED_FRAMES] == np.array(target, np.uint8)).all(-1)
    small = lambda a, dt: np.stack([cv2.resize(x.astype(dt), (size[1], size[0]), interpolation=cv2.INTER_NEAREST) for x in a])
    return {'key': key, 'depth': small(depth, np.float32).astype(np.float16), 'target': small(mask, np.uint8).astype(bool),
            'target_colour': target, 'candidate_moves': {str(k): v for k, v in moves.items()}, 'frames': n}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--embodiment', required=True, choices=['agilex', 'franka', 'agilex_c2r', 'agilex_c2c'])
    p.add_argument('--split', default='artifacts/downstream/split.json')
    p.add_argument('--output', type=Path, default=Path('artifacts/downstream/gt'))
    p.add_argument('--size', type=int, nargs=2, default=[120, 160], help='H W of stored probe targets')
    p.add_argument('--workers', type=int, default=16)
    a = p.parse_args()
    split = load_split(a.split)[a.embodiment]
    keys = split['train'] + split['test']
    tasks = sorted({k.split('__')[0] for k in keys})
    with Pool(a.workers) as pool:
        per_task = pool.map(frame0_colours, [gt_hdf5(a.embodiment, f'{t}__episode0') for t in tasks])
    background = set.intersection(*per_task)
    print(f'{a.embodiment}: {len(tasks)} tasks, background/robot colours at frame 0 in every task: {sorted(background)}', flush=True)
    with Pool(a.workers) as pool:
        res = pool.map(episode_targets, [(a.embodiment, k, background, tuple(a.size)) for k in keys], chunksize=8)
    a.output.mkdir(parents=True, exist_ok=True)
    np.savez(a.output / f'{a.embodiment}.npz', keys=np.array(keys), depth=np.stack([r['depth'] for r in res]),
             target=np.stack([r['target'] for r in res]), task=np.array([k.split('__')[0] for k in keys]),
             probe_aligned_frames=PROBE_ALIGNED_FRAMES)
    audit = {'background_colours': sorted(background), 'min_area': MIN_AREA,
             'episodes': {r['key']: {'target_colour': r['target_colour'], 'candidate_moves': r['candidate_moves'], 'frames': r['frames']} for r in res}}
    (a.output / f'{a.embodiment}_targets.json').write_text(json.dumps(audit, indent=1))
    no_target = [r['key'] for r in res if r['target_colour'] is None]
    print(f'{a.embodiment}: episodes={len(res)} no_target={len(no_target)} {no_target[:5]}', flush=True)


if __name__ == '__main__':
    main()
