#!/usr/bin/env python
"""Per-layer sparse 3D-motion probe on unpooled tokens (the D4RT-style task: query pixel at frame k -> 3D displacement to k+1).

For each layer, a one-block attention read-out whose query is built from the query pixel (u, v) attends over the
tokens of probe frames k and k+1 (each token tagged with its frame) and regresses the 3D displacement of the queried
point (mm, camera frame). Trained on the 10% split (episodes), 60 epochs, scored on the rest.

Steps k = 1..7 are used for every model so that models whose tubelets do not cover frame 0 (VideoMAEv2, D4RT) are
scored on the same samples as the others.

  OpenWAM     latents k, k+1 ->  2 x 80 head tokens x 3072, all 30 layers
  D4RT        tubelets 2k-1, 2k+1 (frames 4k-1,4k / 4k+3,4k+4) -> 2 x 256 x 1280, final encoder memory
  VideoMAEv2  same tubelets -> 2 x 196 x 1024, all 24 layers (tokens recomputed)
  DINOv2      frames 4k, 4k+4 -> 2 x 336 patch tokens x 1024, all 25 layers (tokens recomputed)

Metrics (test set): EPE3D = mean ||pred - gt|| in mm; EPE2D = pixel error of the projected endpoint; zero3d = EPE of
predicting no motion; rel = EPE3D / zero3d; plus EPE3D split by end-effector vs object samples.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_task_cls_layers import Keys, log, openwam_layer_group, split_keys

GT_DIR = Path('artifacts/downstream/gt')
STEPS = range(1, 8)
SCALE = 100.0  # regression target in units of 100 mm


class MotionReadout(nn.Module):
    """Query from (u, v); one cross-attention block over the two frames' tokens; 3-vector output."""

    def __init__(self, dim, n_tokens, width=256, heads=8, n_freq=8):
        super().__init__()
        self.proj = nn.Linear(dim, width)
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, width))
        self.frame = nn.Parameter(torch.zeros(2, 1, width))
        self.freq = 2.0 ** torch.arange(n_freq) * np.pi
        self.q = nn.Sequential(nn.Linear(4 * n_freq, width), nn.GELU(), nn.Linear(width, width))
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width))
        self.head = nn.Linear(width, 3)

    def forward(self, tok_a, tok_b, uv):
        """tok_a/tok_b: (B, n, D) tokens of frame k and k+1 (already standardised); uv: (B, 2) in [0, 1]."""
        kv = torch.cat([self.proj(tok_a) + self.frame[0], self.proj(tok_b) + self.frame[1]], 1) + self.pos
        f = (uv[:, :, None] * self.freq.to(uv.device)).reshape(len(uv), -1)
        q = self.q(torch.cat([torch.sin(f), torch.cos(f)], 1))[:, None]
        z = q + self.attn(q, kv, kv, need_weights=False)[0]
        z = z + self.mlp(self.norm(z))
        return self.head(self.norm(z)).squeeze(1)


def chunk_stats(x, device):
    s1 = torch.zeros(x.shape[-1], device=device, dtype=torch.float64); s2 = torch.zeros_like(s1); n = 0
    for c in range(0, len(x), 32):
        f = torch.as_tensor(x[c:c + 32], device=device).double().reshape(-1, x.shape[-1])
        s1 += f.sum(0); s2 += (f * f).sum(0); n += len(f)
    mean = s1 / n
    return mean.float(), (s2 / n - mean ** 2).clamp_min(0).sqrt().float().clamp_min(1e-4)


def motion_probe(tokens, samples, train_ep, test_ep, K, device, epochs, seed, bs=32):
    """tokens: (N_episodes, 8, n, D) fp16 tokens of probe frames 1..8 (index j = frame 4(j+1)); samples: dict arrays."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    ep, step, kind = samples['episode'], samples['step'], samples['kind']
    uv = samples['uv'] / np.array([320.0, 240.0], np.float32)
    d_gt = (samples['xyz2'] - samples['xyz']).astype(np.float32)
    tr = np.flatnonzero(np.isin(ep, train_ep)); te = np.flatnonzero(np.isin(ep, test_ep))
    mean, std = chunk_stats(tokens[train_ep], device)
    model = MotionReadout(tokens.shape[-1], 2 * tokens.shape[2]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    n_b = int(np.ceil(len(tr) / bs))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * n_b)

    def batch(idx):
        a = torch.as_tensor(tokens[ep[idx], step[idx] - 1], device=device)      # frame 4k   (step k >= 1)
        b = torch.as_tensor(tokens[ep[idx], step[idx]], device=device)          # frame 4k+4
        a = (a.float() - mean) / std; b = (b.float() - mean) / std
        return a, b, torch.as_tensor(uv[idx], device=device), torch.as_tensor(d_gt[idx] / SCALE, device=device)
    model.train()
    for _ in range(epochs):
        for bi in np.array_split(rng.permutation(tr), n_b):
            a, b, q, y = batch(bi)
            loss = F.smooth_l1_loss(model(a, b, q), y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    model.eval(); preds = []
    with torch.no_grad():
        for s in range(0, len(te), 64):
            a, b, q, _ = batch(te[s:s + 64]); preds.append(model(a, b, q).cpu().numpy() * SCALE)
    pred = np.concatenate(preds); gt = d_gt[te]
    e3 = np.linalg.norm(pred - gt, axis=1); zero = np.linalg.norm(gt, axis=1)
    p_end = samples['xyz'][te] + pred
    uv_pred = (K @ p_end.T).T; uv_pred = uv_pred[:, :2] / uv_pred[:, 2:3]
    e2 = np.linalg.norm(uv_pred - samples['uv2'][te], axis=1)
    inview = samples['target_in_view'][te]
    kd = kind[te]
    return dict(epe3d=float(e3.mean()), epe2d=float(e2[inview].mean()), zero3d=float(zero.mean()), rel=float(e3.mean() / zero.mean()),
                epe3d_ee=float(e3[kd < 2].mean()) if (kd < 2).any() else float('nan'), epe3d_obj=float(e3[kd == 2].mean()) if (kd == 2).any() else float('nan'),
                n_test=int(len(te)))


# ---------------- token sources: (N, 8, n, D) for probe frames 4..32 ----------------

def openwam_frames(emb, model, keys, layers, workers):
    flat = openwam_layer_group(emb, model, keys, layers, workers)                  # (N, 720, D) = (latent, h, w)
    return {l: v.reshape(len(keys), 9, 80, -1)[:, 1:] for l, v in flat.items()}


def d4rt_frames(emb, keys):
    out = None
    for i, k in enumerate(keys):
        f = np.load(Path(f'artifacts/{emb}_temporal_features_gt/d4rt/{k}.npz'))['features']   # (16, 16, 16, 1280)
        t = f[1::2].reshape(8, 256, -1)                                                        # tubelets 1,3,..,15 end at frames 4,8,..,32
        if out is None:
            out = np.empty((len(keys),) + t.shape, np.float16)
        out[i] = t
    return {0: out}


def videomae_frames_tokens(frames, layers):
    from probe_task_cls_layers import videomae_layer_group
    flat = videomae_layer_group(frames, layers, keep_tubelets=np.arange(1, 16, 2))                # (N, 8 x 196, 1024): only the 8 tubelets ending at probe frames
    return {l: v.reshape(len(frames), 8, 196, -1) for l, v in flat.items()}


def dinov2_frames_tokens(emb, keys, layers, device):
    from probe_dense_layers import dinov2_dense
    return {l: v.reshape(len(keys), 8, -1, v.shape[-1]) for l, v in dinov2_dense(emb, keys, layers, device).items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--embodiment', required=True, choices=['agilex', 'franka'])
    p.add_argument('--models', nargs='*', default=None)
    p.add_argument('--refs', nargs='*', default=[], choices=['d4rt', 'videomaev2', 'dinov2'])
    p.add_argument('--layers', type=int, nargs='*', default=None)
    p.add_argument('--group', type=int, default=3)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--split', default='artifacts/downstream/split.json')
    p.add_argument('--output', type=Path, default=Path('artifacts/downstream/motion_layers_ep60'))
    p.add_argument('--csv-name', default='layers.csv')
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()

    tr, te = split_keys(a.split, a.embodiment)
    keys = Keys(tr + te, a.embodiment)
    S = np.load(GT_DIR / f'{a.embodiment}_motion.npz')
    pos = {k: i for i, k in enumerate(S['keys'])}
    remap = np.full(len(S['keys']), -1); remap[[pos[k] for k in keys]] = np.arange(len(keys))
    keep = np.isin(S['step'], list(STEPS)) & (remap[S['episode']] >= 0)
    samples = {k: S[k][keep] for k in ('step', 'kind', 'uv', 'xyz', 'xyz2', 'uv2', 'target_in_view')}
    samples['episode'] = remap[S['episode'][keep]]
    train_ep, test_ep = np.arange(len(tr)), np.arange(len(tr), len(keys))
    import h5py
    from downstream_data import gt_hdf5
    with h5py.File(gt_hdf5(a.embodiment, keys[0]), 'r') as f:
        K = f['observation/head_camera/intrinsic_cv'][0].astype(np.float64)
    log(f'{a.embodiment}: {len(samples["step"])} samples ({int((samples["kind"] < 2).sum())} end-effector, {int((samples["kind"] == 2).sum())} object), '
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
                r = motion_probe(x, samples, train_ep, test_ep, K, a.device, a.epochs, a.seed); break
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
        log(f'{model} layer {layer}: epe3d {r["epe3d"]:.1f} mm (zero {r["zero3d"]:.1f}, rel {r["rel"]:.3f}) ee {r["epe3d_ee"]:.1f} obj {r["epe3d_obj"]:.1f} epe2d {r["epe2d"]:.1f}px ({time.time() - t0:.0f}s)')

    def todo(model, n):
        ls = a.layers if a.layers is not None else range(n)
        return [l for l in ls if l < n and (model, l) not in done]

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
