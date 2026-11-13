#!/usr/bin/env python
"""Ground truth for step 3 of wamu/action_read_subspace_plan.md, aligned to the model's video tokens and windows.

Windows: every episode of artifacts/action_read/split.json (fit and eval, ID = clean_50 aligned set, OOD =
randomized_500 replayed set), 3 windows per episode at start s in {0, round((L-33)/2), L-33}, L = head rgb frames of the
official episode (exactly RoboTwinDataset / read_subspace.py).  Video tokens: canvas 384x320 -> grid 12x10 per latent
frame, latent frames 0/1/2 = raw frames s / s+1..s+16 / s+17..s+32 (causal VAE, 4 frames per latent after the first);
GT for a latent frame is taken at its LAST raw frame (s, s+16, s+32).  Head camera = token rows 0-7 (head image
240x320 stretched to 256x320: token row r = head rows 30r..30r+29, column c = 32c..32c+31); rows 8-11 = wrist cameras
(left: cols 0-4, right: cols 5-9; no depth/segmentation GT for them).

Per window (head tokens, latent frames 0..2):  depth (median metres), target_frac (pixels on task target entities),
region (0 target, 1 gripper = fl/fr link6-8, 2 rest of the robot, 3 other moving object, 4 static scene/background,
majority vote), flow3d (latent t -> t+1, camera frame, mm, mean over token pixels), track (token-centre pixel of latent
frame 0 followed to latent frames 1, 2: u, v in head-image pixels + visible).
Per window (scene level, both arms, latent frames 0..2):  link6 pose (xyz + quat), TCP position (link6 + 0.12 m along
local x, RoboTwin get_*_tcp_pose), TCP -> target {A} vector (mm), contact (#fingers on target, grasp) if the contact
replay exists, future TCP trajectory s+1..s+32 relative to s (mm), task id.

Head-token GT needs s+32 < frames of the replayed GT file (132): flagged by head_ok.  Pose-level GT prefers the
full-length contact replay (~/dataspace/openwam-robotwin-contact) and falls back to the 132-frame _extra file (pose_ok).
usage: python scripts/action_read/build_token_gt.py --set ID|OOD [--workers 32]
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, 'scripts')
from robotwin_gt_extra import dense_flow, load_extra, pose_matrix  # noqa: E402

HOME = Path.home()
OFFICIAL = HOME / 'dataspace/openwam/benchmark_data/robotwin2.0/dataset'
GT = {'ID': (HOME / 'dataspace/openwam-robotwin-gt/agilex', 'gt_export'),
      'OOD': (HOME / 'dataspace/openwam-robotwin-gt-random/agilex', 'gt_export_randomized')}
CONTACT = {'ID': HOME / 'dataspace/openwam-robotwin-contact/agilex', 'OOD': HOME / 'dataspace/openwam-robotwin-contact/agilex_random'}
GH, GW, TOK_H, TOK_W = 8, 10, 30, 32
TCP_OFFSET = 0.12
GRIPPER_LINKS = {'fl_link6', 'fl_link7', 'fl_link8', 'fr_link6', 'fr_link7', 'fr_link8'}


def latent_raw_frames(s):
    return [s, s + 16, s + 32]


def target_ids(seg_targets, which=None):
    ids = []
    for k, v in seg_targets.items():
        if which is not None and k != which:
            continue
        if v.get('status') in ('unique', 'mapped'):
            ids += [int(e) for e in v.get('entity_ids', [])]
    return ids


def tcp(pose7):
    M = pose_matrix(pose7)
    return M[:3, 3] + M[:3, 0] * TCP_OFFSET


def token_pool(a, fn):
    """(240, 320[, C]) -> (8, 10[, C]) with fn over each token's 30x32 pixels (fn takes an (n[, C]) array)."""
    out = []
    for r in range(GH):
        row = []
        for c in range(GW):
            row.append(fn(a[r * TOK_H:(r + 1) * TOK_H, c * TOK_W:(c + 1) * TOK_W].reshape(TOK_H * TOK_W, *a.shape[2:])))
        out.append(row)
    return np.array(out)


def one_window(job):
    set_name, task, ep, s, L = job
    root, sub = GT[set_name]
    main = root / task / sub / 'data' / f'episode{ep}.hdf5'
    extra_p = root / task / sub / 'data' / f'episode{ep}_extra.hdf5'
    ex = load_extra(extra_p)
    T_gt = ex['pose'].shape[0]
    fr = latent_raw_frames(s)
    names = list(ex['name']); eids = [int(e) for e in ex['entity_id']]
    idx = {e: i for i, e in enumerate(eids)}
    tgt_all = set(target_ids(ex['targets']))
    tgt_A = target_ids(ex['targets'], '{A}') or sorted(tgt_all)
    out = dict(head_ok=s + 32 < T_gt)

    # ---------------- head tokens
    if out['head_ok']:
        with h5py.File(main, 'r') as m:
            depth = m['observation/head_camera/depth'][fr] / 1000.0
            E = np.vstack([m['observation/head_camera/extrinsic_cv'][s], [0, 0, 0, 1]]).astype(float)
        robot = {e for e, r in zip(eids, ex['is_robot']) if r}
        grip = {eids[i] for i, n in enumerate(names) if n in GRIPPER_LINKS}
        moving = set()
        for e, i in idx.items():
            if e in robot or e in tgt_all:
                continue
            P = ex['pose'][s:s + 33, i]
            if np.linalg.norm(P[:, :3] - P[0, :3], axis=1).max() > 0.005 or (1 - np.abs((P[:, 3:] * P[0, 3:]).sum(1)).min()) > 1.5e-4:
                moving.add(e)                       # > 5 mm or > ~2 deg over the window
        def region_of(ids):
            lab = np.full(ids.shape, 4, np.int8)
            lab[np.isin(ids, list(moving))] = 3
            lab[np.isin(ids, list(robot))] = 2
            lab[np.isin(ids, list(grip))] = 1
            lab[np.isin(ids, list(tgt_all))] = 0
            return lab
        dep, tf, reg = [], [], []
        for k, f in enumerate(fr):
            ids = ex['actor_id'][f].astype(np.int64)
            dep.append(token_pool(depth[k], lambda x: np.median(x[x > 0]) if (x > 0).any() else np.nan))
            tf.append(token_pool(np.isin(ids, list(tgt_all)).astype(np.float32), np.mean))
            reg.append(token_pool(region_of(ids), lambda x: np.bincount(x, minlength=5).argmax()))
        out.update(depth=np.array(dep, np.float32), target_frac=np.array(tf, np.float32), region=np.array(reg, np.int8))
        Rcw = E[:3, :3]
        flow, track, vis = [], [], []
        for t1, t2 in ((fr[0], fr[1]), (fr[1], fr[2])):
            d = dense_flow(main, ex, t1, t2)
            fl = (d['p2'] - d['p1']) @ Rcw.T * 1000.0                          # camera frame, mm
            flow.append(token_pool(fl, lambda x: np.nanmean(x, 0) if np.isfinite(x).all(1).any() else np.full(3, np.nan)))
        for t2 in (fr[1], fr[2]):
            d = dense_flow(main, ex, fr[0], t2)
            cv, cu = np.arange(GH) * TOK_H + TOK_H // 2, np.arange(GW) * TOK_W + TOK_W // 2
            track.append(np.stack([d['u2'][cv][:, cu], d['v2'][cv][:, cu]], -1)); vis.append(d['visible'][cv][:, cu])
        out.update(flow3d=np.array(flow, np.float32), track=np.array(track, np.float32), track_vis=np.array(vis, bool))

    # ---------------- scene level
    # EE: the OFFICIAL endpose (= link6 pose of the official physics, full length, frame-aligned with the model's video).
    var = 'clean_50' if set_name == 'ID' else 'randomized_500'
    with h5py.File(OFFICIAL / task / f'aloha-agilex_{var}' / 'data' / f'episode{ep}.hdf5', 'r') as f:
        endp = np.stack([f['endpose/left_endpose'][:], f['endpose/right_endpose'][:]], 1)          # (L, 2, 7) xyz qw qx qy qz
    out['pose_ok'] = s + 32 < endp.shape[0]
    link6 = endp[fr].astype(np.float32)                                                            # (3, 2, 7)
    tcp_all = lambda f: np.array([tcp(endp[f, a]) for a in range(2)])
    tcp_p = np.array([tcp_all(f) for f in fr])                                                     # (3, 2, 3) m
    fut = np.array([[tcp(endp[s + j, a]) - tcp(endp[s, a]) for j in range(1, 33)] for a in range(2)]) * 1000.0
    out.update(link6=link6, tcp=(tcp_p * 1000.0).astype(np.float32), future_tcp=fut.astype(np.float32))
    # target {A} position: existing replayed GT for frames < 132, the (approximate) contact replay beyond
    cp = CONTACT[set_name] / task / f'episode{ep}_contact.hdf5'
    c = h5py.File(cp, 'r') if cp.exists() else None
    tpos = []
    for f in fr:
        if not tgt_A:
            tpos.append(np.full(3, np.nan))
        elif f < T_gt:
            tpos.append(ex['pose'][f, idx[tgt_A[0]], :3])
        elif c is not None:
            ce = [int(e) for e in c['entities/entity_id'][:]]
            tpos.append(c['entities/pose'][f, ce.index(tgt_A[0]), :3] if f < c['entities/pose'].shape[0] else np.full(3, np.nan))
        else:
            tpos.append(np.full(3, np.nan))
    out['grip2target'] = ((np.array(tpos)[:, None, :] - tcp_p) * 1000.0).astype(np.float32)
    out['target_src_replay'] = np.array([f >= T_gt for f in fr])
    # contact / grasp: approximate SAPIEN replay on this cluster (physics differs from the official machine by mm-cm)
    if c is not None:
        out['n_fingers_on_target'] = np.array([[c[f'contact/{arm}/n_fingers_on_target'][f] for arm in ('left', 'right')] for f in fr], np.int8)
        out['grasp'] = np.array([[c[f'contact/{arm}/grasp'][f] for arm in ('left', 'right')] for f in fr], bool)
        # replay vs existing GT, target {A} position, over the overlapping first min(132, T) frames (m)
        dr = c['drift/target_pos_diff'][:] if 'drift/target_pos_diff' in c else np.array([np.nan])
        out['contact_drift_m'] = np.float32(np.nanmax(dr) if np.isfinite(dr).any() else np.nan)
        c.close()
    return job, out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--set', choices=['ID', 'OOD'], required=True); p.add_argument('--workers', type=int, default=16)
    p.add_argument('--limit', type=int, default=0)
    a = p.parse_args()
    split = json.loads(Path('artifacts/action_read/split.json').read_text())
    sd = split['sets'][a.set]; var = sd['variant']
    tasks = sorted(sd['tasks'])
    jobs = []
    for task, v in sorted(sd['tasks'].items()):
        for half in ('fit', 'eval'):
            for ep in v[half]:
                with h5py.File(OFFICIAL / task / f'aloha-agilex_{var}' / 'data' / f'episode{ep}.hdf5', 'r') as f:
                    L = f['observation/head_camera/rgb'].shape[0]
                ms = max(0, L - 33)
                starts = {int(round(ms * w / 2)) for w in range(3)} | ({s for s in (0, 33, 66, 99) if s <= ms} or {0})
                for st in sorted(starts):
                    jobs.append((a.set, task, ep, st, L))
    halves = {(t, e): h for t, v in sd['tasks'].items() for h in ('fit', 'eval') for e in v[h]}
    if a.limit:
        jobs = jobs[:a.limit]
    print(f'{a.set}: {len(jobs)} windows', flush=True)
    res = []
    with ProcessPoolExecutor(a.workers) as pool:
        for i, r in enumerate(pool.map(one_window, jobs, chunksize=4)):
            res.append(r)
            if (i + 1) % 200 == 0:
                print(f'  {i + 1}/{len(jobs)}', flush=True)
    keys = ['depth', 'target_frac', 'region', 'flow3d', 'track', 'track_vis', 'link6', 'tcp', 'grip2target', 'future_tcp', 'n_fingers_on_target', 'grasp',
            'target_src_replay', 'contact_drift_m']
    shapes = {}
    for _, o in res:
        for k in keys:
            if k in o and k not in shapes:
                shapes[k] = (o[k].shape, o[k].dtype)
    arrs = {k: np.full((len(res), *sh), np.nan if np.issubdtype(dt, np.floating) else (-1 if dt == np.int8 else False), dt) for k, (sh, dt) in shapes.items()}
    for i, (_, o) in enumerate(res):
        for k in shapes:
            if k in o:
                arrs[k][i] = o[k]
    out = Path('artifacts/action_read/gt') / f'{a.set}.npz'; out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, task=np.array([j[1] for j, _ in res]), episode=np.array([j[2] for j, _ in res]), start=np.array([j[3] for j, _ in res]),
                        ep_len=np.array([j[4] for j, _ in res]), half=np.array([halves[(j[1], j[2])] for j, _ in res]),
                        task_id=np.array([tasks.index(j[1]) for j, _ in res]), head_ok=np.array([o['head_ok'] for _, o in res]),
                        pose_ok=np.array([o['pose_ok'] for _, o in res]), **arrs)
    n_head = sum(o['head_ok'] for _, o in res); n_pose = sum(o['pose_ok'] for _, o in res); n_con = sum('grasp' in o for _, o in res)
    print(f'wrote {out}: {len(res)} windows, head GT {n_head}, pose GT {n_pose}, contact {n_con}; fields {sorted(shapes)}', flush=True)


if __name__ == '__main__':
    main()
