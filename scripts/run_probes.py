#!/usr/bin/env python
"""Downstream probes on frozen OpenWAM representations (wamu/downstreamtask.md).

Tasks (train on the 10% split, test on the rest; layers from select_layers_cka.py):
  depth         linear token->log-depth, bilinear upsampling;  AbsRel, delta1, pairwise ordinal accuracy (DA-2K style)
  segmentation  linear token->target-mask logit, upsampling;    target IoU
  classification  task label per episode, following Gen4U's SSv2 protocol: a linear head on the
                spatio-temporal mean and a one-block attention read-out over all tokens; top-1 accuracy.
                Its layer is chosen by 5-fold CV on the train split (CKA vs VideoMAEv2 is flat across layers).
Reference models get the linear classification probe on their (already pooled) features as baselines.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from downstream_data import OPENWAM_ROOT, openwam_layer, load_split

PAIR_MIN_RATIO = 1.1  # pairwise depth: only pairs whose GT depths differ by >= 10%, so the order is unambiguous
PAIRS_PER_FRAME = 200


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def load_tokens(embodiment, model, keys, layer, workers=16):
    with ThreadPoolExecutor(workers) as ex:
        return np.stack(list(ex.map(lambda k: openwam_layer(embodiment, model, k, layer), keys)))  # (N, 9, h, w, D)


class Standardize:
    def __init__(self, x, dim_axis=-1):
        flat = torch.as_tensor(x.reshape(-1, x.shape[dim_axis]), dtype=torch.float32)
        self.mean = flat.mean(0)
        self.std = flat.std(0).clamp_min(1e-4)

    def __call__(self, t):
        return (t.float() - self.mean.to(t.device)) / self.std.to(t.device)


def batches(n, bs, rng=None):
    idx = rng.permutation(n) if rng is not None else np.arange(n)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


def dense_logits(head, norm, x, out_hw, dev):
    """x: (B, L, h, w, D) fp16 numpy -> (B, L, H, W) upsampled per-pixel outputs."""
    t = norm(torch.as_tensor(x, device=dev))
    b, l, h, w, _ = t.shape
    z = head(t).squeeze(-1).reshape(b * l, 1, h, w)
    return F.interpolate(z, size=out_hw, mode='bilinear', align_corners=False).reshape(b, l, *out_hw)


def train_dense(x_tr, y_tr, loss_fn, dev, epochs, seed, lr=1e-3, bs=16, init_bias=0.0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    norm = Standardize(x_tr)
    head = nn.Linear(x_tr.shape[-1], 1).to(dev)
    # Start from the target mean so the probe only has to explain variation (log-depth in mm is ~6.5, not 0).
    nn.init.constant_(head.bias, init_bias)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(x_tr) / bs))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for ep in range(epochs):
        tot = 0.0
        for b in batches(len(x_tr), bs, rng):
            pred = dense_logits(head, norm, x_tr[b], y_tr.shape[-2:], dev)
            loss = loss_fn(pred, torch.as_tensor(y_tr[b], device=dev))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(b)
        if ep in (0, epochs - 1) or (ep + 1) % 10 == 0:
            log(f'    epoch {ep + 1}/{epochs} loss {tot / len(x_tr):.4f}')
    return head.eval(), norm


@torch.no_grad()
def predict_dense(head, norm, x, out_hw, dev, bs=32):
    return np.concatenate([dense_logits(head, norm, x[b], out_hw, dev).float().cpu().numpy() for b in batches(len(x), bs)])


def depth_probe(x_tr, d_tr, x_te, d_te, tasks_te, dev, epochs, seed):
    def to_log(d):
        d = d.astype(np.float32)
        valid = np.isfinite(d) & (d > 0)
        return np.where(valid, np.log(np.where(valid, d, 1.0)), np.nan).astype(np.float32)

    def loss_fn(pred, y):
        m = torch.isfinite(y)
        return (pred[m] - y[m]).abs().mean()

    y_tr = to_log(d_tr)
    head, norm = train_dense(x_tr, y_tr, loss_fn, dev, epochs, seed, init_bias=float(np.nanmean(y_tr)))
    pred = np.exp(predict_dense(head, norm, x_te, d_te.shape[-2:], dev))  # (N, 9, H, W) mm
    gt = d_te.astype(np.float32)
    rng = np.random.default_rng(seed)
    per_frame = []  # (absrel, delta1, pair_acc, task)
    for i in range(len(gt)):
        for l in range(gt.shape[1]):
            g, p = gt[i, l].ravel(), pred[i, l].ravel()
            v = np.isfinite(g) & (g > 0)
            if v.sum() < 2:
                continue
            g, p = g[v], p[v]
            absrel = float(np.mean(np.abs(p - g) / g))
            delta1 = float(np.mean(np.maximum(p / g, g / p) < 1.25))
            a, b = rng.integers(0, len(g), (2, PAIRS_PER_FRAME * 4))
            keep = np.maximum(g[a] / g[b], g[b] / g[a]) >= PAIR_MIN_RATIO
            a, b = a[keep][:PAIRS_PER_FRAME], b[keep][:PAIRS_PER_FRAME]
            pair = float(np.mean(np.sign(p[a] - p[b]) == np.sign(g[a] - g[b]))) if len(a) else np.nan
            per_frame.append((absrel, delta1, pair, tasks_te[i]))
    arr = np.array([r[:3] for r in per_frame], dtype=np.float64)
    per_task = {}
    for t in sorted(set(tasks_te)):
        z = arr[[r[3] == t for r in per_frame]]
        per_task[t] = {'absrel': float(np.nanmean(z[:, 0])), 'delta1': float(np.nanmean(z[:, 1])), 'pairwise_acc': float(np.nanmean(z[:, 2]))}
    return {'absrel': float(np.nanmean(arr[:, 0])), 'delta1': float(np.nanmean(arr[:, 1])), 'pairwise_acc': float(np.nanmean(arr[:, 2])),
            'frames': len(arr), 'pair_min_ratio': PAIR_MIN_RATIO, 'pairs_per_frame': PAIRS_PER_FRAME, 'per_task': per_task}


def seg_probe(x_tr, m_tr, x_te, m_te, tasks_te, dev, epochs, seed):
    pos = float(m_tr.mean())
    pos_weight = torch.tensor(min((1 - pos) / max(pos, 1e-6), 100.0), device=dev)  # targets cover ~1% of the image

    def loss_fn(pred, y):
        return F.binary_cross_entropy_with_logits(pred, y.float(), pos_weight=pos_weight)

    head, norm = train_dense(x_tr, m_tr, loss_fn, dev, epochs, seed)
    pred = predict_dense(head, norm, x_te, m_te.shape[-2:], dev) > 0
    inter = (pred & m_te).sum(axis=(2, 3)).astype(np.float64)
    union = (pred | m_te).sum(axis=(2, 3)).astype(np.float64)
    present = m_te.sum(axis=(2, 3)) > 0
    iou = np.where(present, inter / np.maximum(union, 1), np.nan)  # frames where the target is visible
    per_task = {t: float(np.nanmean(iou[[x == t for x in tasks_te]])) for t in sorted(set(tasks_te))}
    return {'target_iou': float(np.nanmean(iou)), 'global_iou': float(inter[present].sum() / union[present].sum()),
            'frames_with_target': int(present.sum()), 'train_pos_fraction': pos, 'pos_weight': float(pos_weight), 'per_task': per_task}


def fit_linear_classifier(x, y, n_cls, dev, seed, epochs=300, wd=1e-3):
    torch.manual_seed(seed)
    x = torch.as_tensor(x, device=dev, dtype=torch.float32)
    y = torch.as_tensor(y, device=dev)
    mean, std = x.mean(0), x.std(0).clamp_min(1e-4)
    head = nn.Linear(x.shape[1], n_cls).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-2, weight_decay=wd)
    for _ in range(epochs):
        loss = F.cross_entropy(head((x - mean) / std), y)
        opt.zero_grad(); loss.backward(); opt.step()
    return lambda z: head((torch.as_tensor(z, device=dev, dtype=torch.float32) - mean) / std).argmax(1).cpu().numpy()


def cv_accuracy(x, y, n_cls, dev, seed, folds=5):
    """Stratified k-fold accuracy on the train split."""
    rng = np.random.default_rng(seed)
    fold = np.zeros(len(y), int)
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        fold[idx] = np.arange(len(idx)) % folds
    correct = 0
    for f in range(folds):
        tr, va = fold != f, fold == f
        clf = fit_linear_classifier(x[tr], y[tr], n_cls, dev, seed)
        correct += int((clf(x[va]) == y[va]).sum())
    return correct / len(y)


class AttentionReadout(nn.Module):
    """One-block attention read-out: a learnable query cross-attends over all tokens (Gen4U / 4DS protocol)."""

    def __init__(self, dim, n_tokens, n_cls, width=256, heads=8):
        super().__init__()
        self.proj = nn.Linear(dim, width)
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, width))
        self.query = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width))
        self.head = nn.Linear(width, n_cls)

    def forward(self, tokens):
        kv = self.proj(tokens) + self.pos
        q = self.query.expand(len(tokens), -1, -1)
        z = q + self.attn(q, kv, kv, need_weights=False)[0]
        z = z + self.mlp(self.norm(z))
        return self.head(self.norm(z)).squeeze(1)


def attention_probe(x_tr, y_tr, x_te, y_te, n_cls, dev, seed, epochs=150, bs=16):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    norm = Standardize(x_tr)
    flat = lambda x: norm(torch.as_tensor(x, device=dev)).reshape(len(x), -1, x.shape[-1])
    model = AttentionReadout(x_tr.shape[-1], int(np.prod(x_tr.shape[1:-1])), n_cls).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * int(np.ceil(len(x_tr) / bs)))
    for ep in range(epochs):
        model.train()
        for b in batches(len(x_tr), bs, rng):
            loss = F.cross_entropy(model(flat(x_tr[b])), torch.as_tensor(y_tr[b], device=dev))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    model.eval()
    with torch.no_grad():
        pred = np.concatenate([model(flat(x_te[b])).argmax(1).cpu().numpy() for b in batches(len(x_te), 32)])
    return float((pred == y_te).mean())


def ref_episode_vectors(ref_root, ref, keys):
    """Pooled reference features per episode: (N, layers, dim)."""
    out = []
    for k in keys:
        f = np.load(ref_root / ref / f'{k}.npz')['features'].astype(np.float32)
        if ref == 'depthv3':
            f = f.reshape(f.shape[0], -1).mean(0)[None]  # (33, 16, 16) depth maps -> (1, 256)
        elif ref == 'd4rt':
            f = f.mean(axis=(0, 1, 2))[None]
        else:
            f = f.mean(0)  # mean over frames / tubelets -> (layers, dim)
        out.append(f)
    return np.stack(out)


def classify_pooled(x_tr, y_tr, x_te, y_te, n_cls, dev, seed):
    """x: (N, layers, dim). Choose the layer by 5-fold CV on train, then fit on all train."""
    cv = [cv_accuracy(x_tr[:, l], y_tr, n_cls, dev, seed) for l in range(x_tr.shape[1])]
    best = int(np.argmax(cv))
    clf = fit_linear_classifier(x_tr[:, best], y_tr, n_cls, dev, seed)
    return {'layer': best, 'cv_acc_by_layer': cv, 'linear_top1': float((clf(x_te[:, best]) == y_te).mean())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--embodiment', required=True, choices=['agilex', 'franka', 'agilex_c2r', 'agilex_c2c'])
    p.add_argument('--models', nargs='*', default=None, help='default: every OpenWAM model with all episodes extracted')
    p.add_argument('--tasks', nargs='*', default=['depth', 'segmentation', 'classification'])
    p.add_argument('--refs', nargs='*', default=['dinov2', 'videomaev2', 'depthv3', 'd4rt'], help='reference classification baselines')
    p.add_argument('--split', default='artifacts/downstream/split.json')
    p.add_argument('--output', type=Path, default=Path('artifacts/downstream/probes'))
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    dev = torch.device(a.device)
    split = load_split(a.split)[a.embodiment]
    gt = np.load(f'artifacts/downstream/gt/{a.embodiment}.npz')
    index = {k: i for i, k in enumerate(gt['keys'])}
    tr = np.array([index[k] for k in split['train']])
    te = np.array([index[k] for k in split['test']])
    keys_tr, keys_te = list(gt['keys'][tr]), list(gt['keys'][te])
    task_names = sorted(set(gt['task']))
    y = np.array([task_names.index(t) for t in gt['task']])
    out_dir = a.output / a.embodiment
    out_dir.mkdir(parents=True, exist_ok=True)
    sel_dir = Path('artifacts/downstream/layer_selection') / a.embodiment
    selected = json.loads((sel_dir / 'selected_layers.json').read_text())['selected']

    model_root = OPENWAM_ROOT / a.embodiment
    models = a.models or sorted(d.name for d in model_root.iterdir() if d.is_dir() and len(list(d.glob('*.npz'))) == len(index))
    log(f'{a.embodiment}: models={models} train={len(tr)} test={len(te)} classes={len(task_names)}')

    if 'classification' in a.tasks and a.refs and not (out_dir / 'references.json').exists():
        ref_root = Path(f'artifacts/{a.embodiment}_temporal_features_gt')
        refs = {}
        for r in a.refs:
            x = ref_episode_vectors(ref_root, r, list(gt['keys']))
            refs[r] = classify_pooled(x[tr], y[tr], x[te], y[te], len(task_names), dev, a.seed)
            log(f'ref {r}: layer {refs[r]["layer"]} linear top1 {refs[r]["linear_top1"]:.4f}')
        (out_dir / 'references.json').write_text(json.dumps({'classification': refs}, indent=1))

    for m in models:
        res_path = out_dir / f'{m}.json'
        res = json.loads(res_path.read_text()) if res_path.exists() else {}
        if m not in selected and {'depth', 'segmentation'} & set(a.tasks):
            log(f'skip {m}: no selected layers yet (rerun select_layers_cka.py)')
            continue
        layers = {t: selected[m][t]['layer'] for t in ('depth', 'segmentation') if t in a.tasks}
        if 'classification' in a.tasks:
            cache = np.load(sel_dir / 'cache' / f'{m}.npz')
            assert list(cache['keys']) == keys_tr
            pooled_tr = cache['features'].astype(np.float32).mean(1)  # (train, 30, 3072): mean over latents
            cv = [cv_accuracy(pooled_tr[:, l], y[tr], len(task_names), dev, a.seed) for l in range(pooled_tr.shape[1])]
            layers['classification'] = int(np.argmax(cv))
            res['classification_cv_acc_by_layer'] = cv
        log(f'== {m} layers={layers}')
        for layer in sorted(set(layers.values())):
            todo = [t for t, l in layers.items() if l == layer and t not in res]
            if not todo:
                continue
            log(f'  loading layer {layer} for {todo}')
            x = load_tokens(a.embodiment, m, list(gt['keys']), layer)  # (N, 9, 8, 10, D) fp16
            for t in todo:
                log(f'  {t} @ layer {layer}')
                if t == 'depth':
                    r = depth_probe(x[tr], gt['depth'][tr], x[te], gt['depth'][te], list(gt['task'][te]), dev, a.epochs, a.seed)
                    log(f'    absrel {r["absrel"]:.4f} delta1 {r["delta1"]:.4f} pairwise {r["pairwise_acc"]:.4f}')
                elif t == 'segmentation':
                    r = seg_probe(x[tr], gt['target'][tr], x[te], gt['target'][te], list(gt['task'][te]), dev, a.epochs, a.seed)
                    log(f'    target IoU {r["target_iou"]:.4f} global IoU {r["global_iou"]:.4f}')
                else:
                    pooled = x.astype(np.float32).mean(axis=(1, 2, 3))
                    clf = fit_linear_classifier(pooled[tr], y[tr], len(task_names), dev, a.seed)
                    r = {'linear_top1': float((clf(pooled[te]) == y[te]).mean()),
                         'attention_top1': attention_probe(x[tr], y[tr], x[te], y[te], len(task_names), dev, a.seed)}
                    log(f'    linear top1 {r["linear_top1"]:.4f} attention top1 {r["attention_top1"]:.4f}')
                res[t] = {'layer': layer, **r}
                res_path.write_text(json.dumps(res, indent=1))
            del x
        torch.cuda.empty_cache()
    log('done')


if __name__ == '__main__':
    main()
