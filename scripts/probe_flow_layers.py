#!/usr/bin/env python
"""Per-layer dense 3D scene-flow / point-tracking probe on unpooled tokens (the D4RT + TAPVid task).

Query pixel (u, v) at probe frame k; the probe attends over the tokens of frames k and k+1 and predicts the
point's 3D displacement (mm, camera frame) and whether it is still visible at k+1. One probe per layer,
trained on the 10% split (episodes), 60 epochs, scored on the rest. Steps k = 1..7 for every model so that
VideoMAEv2 / D4RT (whose tubelets do not cover frame 0) are scored on the same queries as the others.

Metrics (test set):
  epe3d            mean ||pred - gt|| in mm, also split by group (robot / moving object / static scene)
  zero3d, rel      EPE of predicting no motion, and epe3d / zero3d
  epe2d            pixel error of the projected endpoint on GT-visible queries (320 x 240 frame)
  delta_avg        TAPVid delta^x_avg: mean over thresholds {1,2,4,8,16} px of the fraction of GT-visible queries within threshold
  oa               TAPVid occlusion accuracy: predicted vs GT visibility, all queries
  aj               TAPVid average Jaccard over the same thresholds (TP: predicted visible, GT visible, within threshold)
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_task_cls_layers import _plan_helper, Keys, log, split_keys
from probe_motion_layers import MotionReadout, chunk_stats, d4rt_frames, dinov2_frames_tokens, openwam_frames, videomae_frames_tokens

GT_DIR = Path('artifacts/downstream/gt')
STEPS = range(1, 8)
SCALE = 100.0
THRESH = (1, 2, 4, 8, 16)


class FlowReadout(MotionReadout):
    """MotionReadout with a 4th output: visibility logit at k+1."""

    def __init__(self, dim, n_tokens, **kw):
        super().__init__(dim, n_tokens, **kw)
        self.head = nn.Linear(self.head.in_features, 4)

    def forward_pairs(self, tok_a, tok_b, pair_idx, uv):
        """tok_a/tok_b: (P, n, D) tokens of P (episode, step) pairs; pair_idx: (B,) pair of each query; uv: (B, 2)."""
        kv = torch.cat([self.proj(tok_a) + self.frame[0], self.proj(tok_b) + self.frame[1]], 1) + self.pos   # (P, 2n, W)
        kv = kv[pair_idx]
        f = (uv[:, :, None] * self.freq.to(uv.device)).reshape(len(uv), -1)
        q = self.q(torch.cat([torch.sin(f), torch.cos(f)], 1))[:, None]
        z = q + self.attn(q, kv, kv, need_weights=False)[0]
        z = z + self.mlp(self.norm(z))
        return self.head(self.norm(z)).squeeze(1)


def flow_probe(tokens, S, train_ep, test_ep, K, device, epochs, seed, pairs_per_batch=32):
    """tokens: (N_episodes, 8, n, D) fp16 for probe frames 4..32; S: sample arrays (episode remapped, steps 1..7)."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    ep, step = S['episode'], S['step']
    uv = S['uv'] / np.array([320.0, 240.0], np.float32)
    dxyz, vis = S['dxyz'].astype(np.float32), S['visible'].astype(np.float32)
    pair_id = ep * 8 + step
    # group query rows by (episode, step) so each token pair is moved to the GPU once per batch
    order = np.argsort(pair_id, kind='stable'); pid_sorted = pair_id[order]
    uniq, starts = np.unique(pid_sorted, return_index=True); ends = np.append(starts[1:], len(order))
    rows_of = {int(p): order[s:e] for p, s, e in zip(uniq, starts, ends)}
    tr_pairs = np.array([p for p in uniq if (p // 8) in set(train_ep.tolist())]); te_pairs = np.array([p for p in uniq if (p // 8) in set(test_ep.tolist())])
    mean, std = chunk_stats(tokens[train_ep], device)
    model = FlowReadout(tokens.shape[-1], 2 * tokens.shape[2]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05)   # 32 pairs (~1.5k queries) per step
    n_b = int(np.ceil(len(tr_pairs) / pairs_per_batch))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * n_b)

    def batch(pairs):
        e, s = pairs // 8, pairs % 8
        a = (torch.as_tensor(tokens[e, s - 1], device=device).float() - mean) / std
        b = (torch.as_tensor(tokens[e, s], device=device).float() - mean) / std
        rows = np.concatenate([rows_of[int(p)] for p in pairs]); pidx = np.concatenate([[i] * len(rows_of[int(p)]) for i, p in enumerate(pairs)])
        return a, b, torch.as_tensor(pidx, device=device), torch.as_tensor(uv[rows], device=device), rows
    model.train()
    for _ in range(epochs):
        for pb in np.array_split(rng.permutation(tr_pairs), n_b):
            a, b, pidx, q, rows = batch(pb)
            out = model.forward_pairs(a, b, pidx, q)
            loss = F.smooth_l1_loss(out[:, :3], torch.as_tensor(dxyz[rows] / SCALE, device=device)) \
                + 0.1 * F.binary_cross_entropy_with_logits(out[:, 3], torch.as_tensor(vis[rows], device=device))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    model.eval(); preds, idx = [], []
    with torch.no_grad():
        for pb in np.array_split(te_pairs, max(1, len(te_pairs) // 32)):
            a, b, pidx, q, rows = batch(pb); preds.append(model.forward_pairs(a, b, pidx, q).cpu().numpy()); idx.append(rows)
    out = np.concatenate(preds); te = np.concatenate(idx)
    pred = out[:, :3] * SCALE; pvis = out[:, 3] > 0
    gt = dxyz[te]; gvis = S['visible'][te]; grp = S['group'][te]
    e3 = np.linalg.norm(pred - gt, axis=1); zero = np.linalg.norm(gt, axis=1)
    # 2D endpoint: project the GT camera-frame start point moved by the predicted displacement.
    p1 = S['p1'][te]
    pe = p1 + pred; uvp = (K @ pe.T).T; uvp = uvp[:, :2] / np.clip(uvp[:, 2:3], 1e-6, None) - 0.5   # dense_flow's pixel-centre convention
    e2 = np.linalg.norm(uvp - S['uv2'][te], axis=1)
    within = {t: e2 < t for t in THRESH}
    delta = float(np.mean([within[t][gvis].mean() for t in THRESH]))
    oa = float((pvis == gvis).mean())
    aj = []
    for t in THRESH:
        tp = (pvis & gvis & within[t]).sum(); fp = (pvis & ~(gvis & within[t])).sum(); fn = (gvis & ~(pvis & within[t])).sum()
        aj.append(tp / max(tp + fp + fn, 1))
    r = dict(epe3d=float(e3.mean()), zero3d=float(zero.mean()), rel=float(e3.mean() / zero.mean()),
             epe3d_robot=float(e3[grp == 0].mean()), epe3d_moving=float(e3[grp == 1].mean()), epe3d_static=float(e3[grp == 2].mean()),
             epe2d=float(e2[gvis].mean()), delta_avg=delta, oa=oa, aj=float(np.mean(aj)), n_test=int(len(te)))
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--embodiment', required=True, choices=['agilex', 'franka'])
    p.add_argument('--models', nargs='*', default=None)
    p.add_argument('--refs', nargs='*', default=[], choices=['d4rt', 'videomaev2', 'dinov2'])
    p.add_argument('--layers', type=int, nargs='*', default=None)
    p.add_argument('--group', type=int, default=3)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--split', default='artifacts/downstream/split.json')
    p.add_argument('--output', type=Path, default=Path('artifacts/downstream/flow_layers_ep60'))
    p.add_argument('--csv-name', default='layers.csv')
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda')
    p.add_argument('--layer-plan', type=Path, default=None, help='JSON {embodiment: {task: {model: [layers]}}}: probe only these layers per model (e.g. the top-10 of a previous seed)')
    a = p.parse_args()

    tr, te = split_keys(a.split, a.embodiment)
    planned = _plan_helper(a, 'flow')
    keys = Keys(tr + te, a.embodiment)
    G = np.load(GT_DIR / f'{a.embodiment}_flow.npz')
    pos = {k: i for i, k in enumerate(G['keys'])}
    remap = np.full(len(G['keys']), -1); remap[[pos[k] for k in keys]] = np.arange(len(keys))
    keep = np.isin(G['step'], list(STEPS)) & (remap[G['episode']] >= 0)
    S = {k: G[k][keep] for k in ('step', 'group', 'uv', 'dxyz', 'uv2', 'visible')}
    S['episode'] = remap[G['episode'][keep]]
    S['p1'] = G['p1'][keep]
    train_ep, test_ep = np.arange(len(tr)), np.arange(len(tr), len(keys))
    from downstream_data import gt_hdf5
    with h5py.File(gt_hdf5(a.embodiment, keys[0]), 'r') as f:
        K = f['observation/head_camera/intrinsic_cv'][0].astype(np.float64)
    log(f'{a.embodiment}: {len(S["step"])} queries (robot/moving/static: {[int((S["group"] == g).sum()) for g in range(3)]}), '
        f'{len(tr)} train / {len(te)} test episodes, {a.epochs} epochs')

    out_dir = a.output / a.embodiment; out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / a.csv_name
    rows = list(csv.DictReader(open(csv_path))) if csv_path.exists() else []
    done = {(r['model'], int(r['layer'])) for r in rows}

    def run(model, kind, layer, x):
        if (model, layer) in done:
            return
        t0 = time.time()
        for attempt in range(10):
            try:
                r = flow_probe(x, S, train_ep, test_ep, K, a.device, a.epochs, a.seed); break
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                if attempt == 9:
                    raise
                log('  CUDA OOM, retry in 60s'); time.sleep(60)
        row = dict(model=model, kind=kind, layer=layer, tokens=int(2 * x.shape[2]), **{k: (f'{v:.4f}' if isinstance(v, float) else v) for k, v in r.items()}, epochs=a.epochs)
        rows.append(row)
        with open(csv_path, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(row)); w.writeheader(); w.writerows(rows)
        torch.cuda.empty_cache()
        log(f'{model} layer {layer}: epe3d {r["epe3d"]:.1f}mm (zero {r["zero3d"]:.1f}, rel {r["rel"]:.3f}; robot {r["epe3d_robot"]:.1f} moving {r["epe3d_moving"]:.1f} static {r["epe3d_static"]:.1f}) '
            f'epe2d {r["epe2d"]:.1f}px delta {r["delta_avg"]:.3f} oa {r["oa"]:.3f} aj {r["aj"]:.3f} ({time.time() - t0:.0f}s)')

    def todo(model, n):
        return [l for l in planned(model, n) if (model, l) not in done]

    for ref in a.refs:
        if ref == 'd4rt' and todo('d4rt', 1):
            run('d4rt', 'reference', 0, d4rt_frames(a.embodiment, keys)[0])
        elif ref == 'videomaev2':
            ls = todo('videomaev2', 24)
            if ls:
                from probe_task_cls_layers import videomae_frames
                frames = videomae_frames(keys, a.workers)
                for g in range(0, len(ls), a.group):
                    xs = videomae_frames_tokens(frames, ls[g:g + a.group])
                    for l in ls[g:g + a.group]:
                        run('videomaev2', 'reference', l, xs[l])
                    del xs
        elif ref == 'dinov2':
            ls = todo('dinov2', 25)
            for g in range(0, len(ls), a.group):
                xs = dinov2_frames_tokens(a.embodiment, keys, ls[g:g + a.group], a.device)
                for l in ls[g:g + a.group]:
                    run('dinov2', 'reference', l, xs[l])
                del xs

    from downstream_data import OPENWAM_ROOT
    models = a.models if a.models is not None else sorted(d.name for d in (OPENWAM_ROOT / a.embodiment).iterdir() if all((d / f'{k}.npz').exists() for k in keys))
    for m in models:
        ls = todo(m, 30)
        for g in range(0, len(ls), a.group):
            xs = openwam_frames(a.embodiment, m, keys, ls[g:g + a.group], a.workers)
            for l in ls[g:g + a.group]:
                run(m, 'openwam', l, xs[l])
            del xs


if __name__ == '__main__':
    main()
