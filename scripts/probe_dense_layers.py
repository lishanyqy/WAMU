#!/usr/bin/env python
"""Per-layer dense downstream probes on unpooled tokens: depth and target-object segmentation.

For every layer of every model, a linear head maps each token to one value; the token grid is
bilinearly upsampled to the 120 x 160 GT (run_probes.py's dense head, unchanged). Trained on the 10%
split (artifacts/downstream/split.json), scored on the rest, few-shot: --epochs (default 10).

Scored frames: aligned frames 4, 8, ..., 32, i.e. the last frame of OpenWAM latents 1-8. VideoMAEv2 and
D4RT tubelets (aligned frames 1-32 as native clips) cover exactly those frames, so every model is
scored on the same 8 frames; frame 0 (OpenWAM latent 0 only) is left out.

  OpenWAM     latents 1-8, 8 x 10 head tokens x 3072, 30 layers   (stored tokens)
  VideoMAEv2  tubelets 1,3,..,15 (frames 4k-1, 4k), 14 x 14 x 1024, 24 layers  (recomputed)
  D4RT        same tubelets, 16 x 16 x 1280, final encoder memory              (stored tokens)
  DINOv2      frames 4k, 16 x 21 patch tokens x 1024, 25 layers; full 240 x 320 frame resized to
              224 x 294 (no centre crop, so patches stay aligned with the GT)    (recomputed)

Metrics: depth AbsRel, delta1, pairwise ordinal accuracy (pairs whose GT depths differ by >= 10%,
DA-2K style); segmentation IoU on frames where the target is visible. All are per-frame means.
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
from downstream_data import npz_member_memmap
from probe_task_cls_layers import Keys, _plan_helper, log, openwam_layer_group, split_keys

GT_DIR = Path('artifacts/downstream/gt')
LATENTS = slice(1, 9)                    # OpenWAM latents 1..8 -> aligned frames 4..32
TUBELETS = np.arange(1, 16, 2)           # tubelet j covers aligned frames 2j+1, 2j+2 -> frames 4k-1, 4k
PAIR_MIN_RATIO, PAIRS_PER_FRAME = 1.1, 200


# ---------------- dense head (as scripts/run_probes.py) ----------------

def stats(x):
    """Per-channel mean/std over every train token, in chunks (fp16 numpy (N, L, h, w, D))."""
    d = x.shape[-1]
    s1 = np.zeros(d, np.float64); s2 = np.zeros(d, np.float64); n = 0
    for c in range(0, len(x), 16):
        f = x[c:c + 16].reshape(-1, d).astype(np.float64)
        s1 += f.sum(0); s2 += (f * f).sum(0); n += len(f)
    mean = s1 / n
    return torch.tensor(mean, dtype=torch.float32), torch.tensor(np.sqrt(np.maximum(s2 / n - mean ** 2, 0)), dtype=torch.float32).clamp_min(1e-4)


def dense_out(head, mean, std, x, out_hw, dev):
    t = (torch.as_tensor(x, device=dev).float() - mean) / std
    b, l, h, w, _ = t.shape
    z = head(t).squeeze(-1).reshape(b * l, 1, h, w)
    return F.interpolate(z, size=out_hw, mode='bilinear', align_corners=False).reshape(b, l, *out_hw)


def train_dense(x_tr, y_tr, loss_fn, dev, epochs, seed, lr=1e-3, bs=16, init_bias=0.0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mean, std = (t.to(dev) for t in stats(x_tr))
    head = nn.Linear(x_tr.shape[-1], 1).to(dev)
    nn.init.constant_(head.bias, init_bias)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    n_b = int(np.ceil(len(x_tr) / bs))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * n_b)
    yt = torch.as_tensor(y_tr, device=dev)
    for _ in range(epochs):
        for b in np.array_split(rng.permutation(len(x_tr)), n_b):
            loss = loss_fn(dense_out(head, mean, std, x_tr[b], y_tr.shape[-2:], dev), yt[b])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return head.eval(), mean, std


@torch.no_grad()
def predict(head, mean, std, x, out_hw, dev, bs=16):
    return np.concatenate([dense_out(head, mean, std, x[s:s + bs], out_hw, dev).cpu().numpy() for s in range(0, len(x), bs)])


# ---------------- tasks ----------------

def depth_probe(x_tr, d_tr, x_te, d_te, dev, epochs, seed):
    def to_log(d):
        d = d.astype(np.float32)
        return np.where(np.isfinite(d) & (d > 0), np.log(np.where(d > 0, d, 1.0)), np.nan).astype(np.float32)

    def loss_fn(pred, y):
        m = torch.isfinite(y)
        return (pred[m] - y[m]).abs().mean()
    y_tr = to_log(d_tr)
    head, mean, std = train_dense(x_tr, y_tr, loss_fn, dev, epochs, seed, init_bias=float(np.nanmean(y_tr)))
    p = np.exp(predict(head, mean, std, x_te, d_te.shape[-2:], dev)).reshape(-1, d_te.shape[-2] * d_te.shape[-1])
    g = d_te.astype(np.float32).reshape(len(p), -1)
    v = np.isfinite(g) & (g > 0)
    gs = np.where(v, g, 1.0)
    keep = v.sum(1) >= 2
    absrel = np.where(v, np.abs(p - gs) / gs, 0).sum(1) / np.maximum(v.sum(1), 1)
    delta1 = np.where(v, np.maximum(p / gs, gs / p) < 1.25, 0).sum(1) / np.maximum(v.sum(1), 1)
    rng = np.random.default_rng(seed)
    a, b = rng.integers(0, g.shape[1], (2, len(g), PAIRS_PER_FRAME * 4))
    rows = np.arange(len(g))[:, None]
    ok = v[rows, a] & v[rows, b] & (np.maximum(gs[rows, a] / gs[rows, b], gs[rows, b] / gs[rows, a]) >= PAIR_MIN_RATIO)
    ok &= np.cumsum(ok, 1) <= PAIRS_PER_FRAME
    right = (np.sign(p[rows, a] - p[rows, b]) == np.sign(g[rows, a] - g[rows, b])) & ok
    pair = np.where(ok.sum(1) > 0, right.sum(1) / np.maximum(ok.sum(1), 1), np.nan)
    return dict(absrel=float(absrel[keep].mean()), delta1=float(delta1[keep].mean()), pairwise=float(np.nanmean(pair[keep])))


def seg_probe(x_tr, m_tr, x_te, m_te, dev, epochs, seed):
    pos = float(m_tr.mean())
    pos_weight = torch.tensor(min((1 - pos) / max(pos, 1e-6), 100.0), device=dev)

    def loss_fn(pred, y):
        return F.binary_cross_entropy_with_logits(pred, y.float(), pos_weight=pos_weight)
    head, mean, std = train_dense(x_tr, m_tr, loss_fn, dev, epochs, seed)
    pred = predict(head, mean, std, x_te, m_te.shape[-2:], dev) > 0
    inter = (pred & m_te).sum((2, 3)).astype(np.float64)
    union = (pred | m_te).sum((2, 3)).astype(np.float64)
    present = m_te.sum((2, 3)) > 0
    return dict(iou=float((inter / np.maximum(union, 1))[present].mean()))


# ---------------- token sources: {layer: (N, 8, h, w, D) float16} ----------------

def openwam_dense(embodiment, model, keys, layers, workers):
    flat = openwam_layer_group(embodiment, model, keys, layers, workers)   # (N, 720, D) in (latent, h, w) order
    return {l: v.reshape(len(keys), 9, 8, 10, -1)[:, LATENTS] for l, v in flat.items()}


def d4rt_dense(embodiment, keys):
    out = None
    for i, k in enumerate(keys):
        f = np.load(Path(f'artifacts/{embodiment}_temporal_features_gt/d4rt/{k}.npz'))['features'][TUBELETS]  # (8, 16, 16, 1280)
        if out is None:
            out = np.empty((len(keys),) + f.shape, np.float16)
        out[i] = f
    return {0: out}


def videomae_dense(frames, layers):
    from probe_task_cls_layers import videomae_layer_group
    flat = videomae_layer_group(frames, layers)                           # (N, 3136, 1024): 16 tubelets x 14 x 14
    return {l: v.reshape(len(frames), 16, 14, 14, -1)[:, TUBELETS] for l, v in flat.items()}


def _stacked_ref(path_of, keys, member, layer_axis, layers):
    """{layer: (N, 8, h, w, D)} from per-episode npz files whose `member` is (L, 8, h, w, D) (or (8, h, w, D) when layer_axis is None)."""
    out = {}
    for i, k in enumerate(keys):
        path = path_of(k)
        z = npz_member_memmap(path, member)
        for l in layers:
            # one contiguous read per slice (NFS: page-faulting through the memmap is ~10x slower than a sequential read)
            sub = z[l] if layer_axis is not None else z
            with open(path, 'rb') as fh:
                fh.seek(z.offset + (l * z.strides[0] if layer_axis is not None else 0))
                x = np.frombuffer(fh.read(sub.nbytes), dtype=z.dtype).reshape(sub.shape)
            if l not in out:
                out[l] = np.empty((len(keys),) + x.shape, np.float16)
            out[l][i] = x
        if (i + 1) % 500 == 0:
            log(f'  read {member} layers {layers}: {i + 1}/{len(keys)}')
    return out


def sam_dense(embodiment, keys, layers):
    """SAM ViT-H tokens (scripts/extract_robotwin_sam_features.py): layers 0-3 = encoder blocks 8/16/24/32 (1280-d),
    layer 4 = the 256-d neck image embedding; all 16 x 16."""
    src = Path(f'artifacts/{embodiment}_features_gt/sam')
    out = {}
    vit = [l for l in layers if l < 4]
    if vit:
        out.update(_stacked_ref(lambda k: src / f'{k}.npz', keys, 'vit', 0, vit))
    if 4 in layers:
        out[4] = _stacked_ref(lambda k: src / f'{k}.npz', keys, 'neck', None, [4])[4]
    return out


def da3_dense(embodiment, keys, layers):
    """Depth Anything 3 backbone tokens (scripts/extract_robotwin_da3_features.py): layers 0-4 = ViT-B blocks 3/5/7/9/11, 12 x 16 x 768."""
    src = Path(f'artifacts/{embodiment}_features_gt/da3')
    return _stacked_ref(lambda k: src / f'{k}.npz', keys, 'features', 0, layers)


_DINO = {}


def dinov2_dense(embodiment, keys, layers, device='cuda', size=(224, 294)):
    """Patch tokens of the scored frames (aligned 4k), all requested layers, no centre crop."""
    from transformers import AutoModel
    from robotwin_gt_frames import load_row_frames
    from downstream_data import gt_hdf5
    if 'm' not in _DINO:
        _DINO['m'] = AutoModel.from_pretrained('facebook/dinov2-large', torch_dtype=torch.float16).to(device).eval()
    model = _DINO['m']
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    gh, gw = size[0] // 14, size[1] // 14
    out = {}
    with torch.inference_mode():
        for i, k in enumerate(keys):
            frames = load_row_frames({'gt_hdf5': str(gt_hdf5(embodiment, k))})[1]
            x = torch.stack([torch.as_tensor(np.asarray(frames[4 * j])) for j in range(1, 9)]).to(device).permute(0, 3, 1, 2).float() / 255
            x = F.interpolate(x, size=size, mode='bilinear', align_corners=False, antialias=True)
            hs = model(pixel_values=((x - mean) / std).half(), output_hidden_states=True).hidden_states
            for l in layers:
                t = hs[l][:, 1:].reshape(8, gh, gw, -1).cpu().numpy()          # drop CLS (no register tokens in dinov2-large)
                if l not in out:
                    out[l] = np.empty((len(keys),) + t.shape, np.float16)
                out[l][i] = t
            if (i + 1) % 500 == 0:
                log(f'  dinov2 layers {layers[0]}-{layers[-1]}: {i + 1}/{len(keys)}')
    return out


# ---------------- driver ----------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--embodiment', required=True, choices=['agilex', 'franka', 'agilex_c2r', 'agilex_c2c'])
    p.add_argument('--models', nargs='*', default=None, help='OpenWAM models (default: all with every split episode extracted)')
    p.add_argument('--refs', nargs='*', default=[], choices=['d4rt', 'videomaev2', 'dinov2', 'sam', 'da3'])
    p.add_argument('--layers', type=int, nargs='*', default=None)
    p.add_argument('--group', type=int, default=5)
    p.add_argument('--tasks', nargs='*', default=['depth', 'segmentation'])
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--split', default='artifacts/downstream/split.json')
    p.add_argument('--output', type=Path, default=Path('artifacts/downstream/dense_layers_ep10'))
    p.add_argument('--csv-name', default='layers.csv')
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda')
    p.add_argument('--layer-plan', type=Path, default=None, help='JSON {embodiment: {task: {model: [layers]}}}: probe only these layers per model (e.g. the top-10 of a previous seed)')
    a = p.parse_args()

    tr, te = split_keys(a.split, a.embodiment)
    TASK_KEY = {'depth': 'depth', 'segmentation': 'seg'}
    planned = {t: _plan_helper(a, TASK_KEY[t]) for t in a.tasks}
    keys = Keys(tr + te, a.embodiment)
    n_tr = len(tr)
    gt = np.load(GT_DIR / f'{a.embodiment}.npz')
    pos = {k: i for i, k in enumerate(gt['keys'])}
    idx = [pos[k] for k in keys]
    depth = gt['depth'][idx][:, LATENTS]          # (N, 8, 120, 160) mm, aligned frames 4..32
    target = gt['target'][idx][:, LATENTS]
    assert list(gt['probe_aligned_frames'][LATENTS]) == [4 * k for k in range(1, 9)]
    log(f'{a.embodiment}: {n_tr} train / {len(te)} test episodes x 8 frames, tasks {a.tasks}, {a.epochs} epochs')

    out_dir = a.output / a.embodiment
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / a.csv_name
    rows = list(csv.DictReader(open(csv_path))) if csv_path.exists() else []
    done = {(r['model'], int(r['layer']), r['task']) for r in rows}

    def run(model, kind, layer, x):
        for task in a.tasks:
            if (model, layer, task) in done or layer not in planned[task](model, 10 ** 6):
                continue
            t0 = time.time()
            for attempt in range(10):
                try:
                    if task == 'depth':
                        r = depth_probe(x[:n_tr], depth[:n_tr], x[n_tr:], depth[n_tr:], a.device, a.epochs, a.seed)
                    else:
                        r = seg_probe(x[:n_tr], target[:n_tr], x[n_tr:], target[n_tr:], a.device, a.epochs, a.seed)
                    break
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if attempt == 9:
                        raise
                    log('  CUDA OOM, retry in 60s'); time.sleep(60)
            row = dict(model=model, kind=kind, layer=layer, task=task, tokens=int(np.prod(x.shape[2:4])),
                       absrel=r.get('absrel', ''), delta1=r.get('delta1', ''), pairwise=r.get('pairwise', ''), iou=r.get('iou', ''), epochs=a.epochs)
            rows.append(row)
            with open(csv_path, 'w', newline='') as fh:
                w = csv.DictWriter(fh, fieldnames=list(row)); w.writeheader(); w.writerows(rows)
            torch.cuda.empty_cache()
            log(f'{model} layer {layer} {task}: ' + ' '.join(f'{k} {v:.4f}' for k, v in r.items()) + f' ({time.time() - t0:.0f}s)')

    def todo(model, n_layers):
        return sorted({l for t in a.tasks for l in planned[t](model, n_layers) if (model, l, t) not in done})

    for ref in a.refs:
        if ref == 'd4rt':
            if todo('d4rt', 1):
                run('d4rt', 'reference', 0, d4rt_dense(a.embodiment, keys)[0])
        elif ref == 'videomaev2':
            ls = todo('videomaev2', 24)
            if ls:
                from probe_task_cls_layers import videomae_frames
                frames = videomae_frames(keys, a.workers)
                for g in range(0, len(ls), a.group):
                    xs = videomae_dense(frames, ls[g:g + a.group])
                    for l in ls[g:g + a.group]:
                        run('videomaev2', 'reference', l, xs[l])
                    del xs
        elif ref == 'dinov2':
            ls = todo('dinov2', 25)
            for g in range(0, len(ls), a.group):
                xs = dinov2_dense(a.embodiment, keys, ls[g:g + a.group], a.device)
                for l in ls[g:g + a.group]:
                    run('dinov2', 'reference', l, xs[l])
                del xs
        elif ref in ('sam', 'da3'):
            ls = todo(ref, 5)
            for l in ls:
                xs = (sam_dense if ref == 'sam' else da3_dense)(a.embodiment, keys, [l])
                run(ref, 'reference', l, xs[l])
                del xs

    from downstream_data import OPENWAM_ROOT
    models = a.models
    if models is None:
        models = sorted(d.name for d in (OPENWAM_ROOT / a.embodiment).iterdir() if all((d / f'{k}.npz').exists() for k in keys))
    for m in models:
        ls = todo(m, 30)
        for g in range(0, len(ls), a.group):
            xs = openwam_dense(a.embodiment, m, keys, ls[g:g + a.group], a.workers)
            for l in ls[g:g + a.group]:
                run(m, 'openwam', l, xs[l])
            del xs


if __name__ == '__main__':
    main()
