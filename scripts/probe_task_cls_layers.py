#!/usr/bin/env python
"""Per-layer task-classification probe on unpooled tokens (downstream capability, not similarity).

For every OpenWAM DiT layer, a one-block attention read-out (Gen4U / 4DS protocol: a learnable query
cross-attends over every token of that layer) is trained on the 10% split
(artifacts/downstream/split.json, 5 episodes per task) and scored on the remaining 90%.
Labels are the Robotwin task names (Agilex 50-way, Franka 46-way). Nothing is pooled before the probe:

  OpenWAM   (9 latents, 8 x 10 head tokens) = 720 tokens x 3072 per layer, all 30 layers
  D4RT      (16 tubelets, 16 x 16)          = 4096 tokens x 1280, final encoder memory
  DINOv2    33 per-frame CLS tokens x 1024 per layer (stored features keep only CLS)

  VideoMAEv2 (16 tubelets, 14 x 14)         = 3136 tokens x 1024 per layer, all 24 layers
VideoMAEv2's stored features are averaged over space, so its tokens are recomputed here on the fly
(aligned frames 1-16 and 17-32 as two native clips, exactly as extract_robotwin_videomaev2_features.py)
instead of being stored (~740 GB). Run that part in the `videomae` env, which has transformers.
"""
import argparse
import csv
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
from downstream_data import OPENWAM_ROOT, npz_member_memmap

TOKEN_REFS = ('d4rt', 'dinov2')
POOLED_REFS = ()
VIDEOMAE_DIR = Path('/home/a1905721/modelspace/VideoMAEv2-Large')


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def _plan_helper(a, task):
    """planned(model, n_layers): the layers to probe for one model under --layer-plan / --layers / all."""
    plan = None
    if a.layer_plan is not None:
        plan = json.loads(Path(a.layer_plan).read_text())[a.embodiment][task]

    def planned(model, n):
        if plan is not None:
            return [l for l in plan.get(model, []) if l < n]
        return [l for l in (a.layers if a.layers is not None else range(n)) if l < n]
    return planned


def split_keys(split, embodiment):
    s = json.loads(Path(split).read_text())[embodiment]
    return s['train'], s['test']


class AttentionReadout(nn.Module):
    """One-block attention read-out: a learnable query cross-attends over all tokens."""

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


def macro_ovr_auroc(y, prob):
    """Macro one-vs-rest AUROC (Mann-Whitney U with average ranks), as sklearn's roc_auc_score(multi_class='ovr')."""
    from scipy.stats import rankdata
    aucs = []
    for c in range(prob.shape[1]):
        pos = y == c
        if pos.all() or not pos.any():
            continue
        r = rankdata(prob[:, c])
        n_pos, n_neg = pos.sum(), (~pos).sum()
        aucs.append((r[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    return float(np.mean(aucs))


def attention_probe(x_tr, y_tr, x_te, y_te, n_cls, seed, epochs, bs=16, device='cuda'):
    """x_*: (N, tokens, dim) float16 numpy. Per-channel standardisation from train tokens."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    xt = torch.as_tensor(x_tr, device=device)                       # fp16 on GPU
    # Per-channel stats in chunks: a full fp32 copy of D4RT's 4096-token train set alone is ~5 GB.
    s1 = torch.zeros(xt.shape[-1], device=device, dtype=torch.float64)
    s2 = torch.zeros_like(s1)
    for c in range(0, len(xt), 16):
        chunk = xt[c:c + 16].double()
        s1 += chunk.sum((0, 1)); s2 += (chunk * chunk).sum((0, 1))
    n_tok = xt.shape[0] * xt.shape[1]
    mean = (s1 / n_tok).float()
    std = (s2 / n_tok - (s1 / n_tok) ** 2).clamp_min(0).sqrt().float().clamp_min(1e-4)
    norm = lambda x: ((x.float() - mean) / std)
    yt = torch.as_tensor(y_tr, device=device)
    model = AttentionReadout(x_tr.shape[-1], x_tr.shape[1], n_cls).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    steps = epochs * int(np.ceil(len(x_tr) / bs))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    model.train()
    for _ in range(epochs):
        for b in np.array_split(rng.permutation(len(x_tr)), int(np.ceil(len(x_tr) / bs))):
            loss = F.cross_entropy(model(norm(xt[b])), yt[b])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    model.eval()
    probs = []
    with torch.no_grad():
        for s in range(0, len(x_te), 16):
            probs.append(torch.softmax(model(norm(torch.as_tensor(x_te[s:s + 16], device=device))), 1).cpu().numpy())
    prob = np.concatenate(probs)
    acc = float((prob.argmax(1) == y_te).mean())
    auroc = macro_ovr_auroc(y_te, prob)
    del xt
    torch.cuda.empty_cache()
    return acc, auroc


def probe_with_retry(*args, tries=10, wait=60, **kw):
    """The GPU is shared with the extraction job and other users; wait out a transient OOM instead of dying."""
    for t in range(tries):
        try:
            return attention_probe(*args, **kw)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if t == tries - 1:
                raise
            log(f'  CUDA OOM, retry {t + 1}/{tries - 1} in {wait}s')
            time.sleep(wait)


def openwam_layer_group(embodiment, model, keys, layers, workers):
    """{layer: (N, 720, 3072) float16} read straight from the token files (memory-mapped slices)."""
    src = OPENWAM_ROOT / embodiment / model
    l0, l1 = layers[0], layers[-1] + 1
    shape = npz_member_memmap(src / f'{keys[0]}.npz').shape          # (9, 30, 8, 10, 3072)
    n_tok = shape[0] * shape[2] * shape[3]
    out = {l: np.empty((len(keys), n_tok, shape[-1]), np.float16) for l in layers}

    def one(i):
        x = np.asarray(npz_member_memmap(src / f'{keys[i]}.npz')[:, l0:l1])  # (9, G, 8, 10, D)
        for j, l in enumerate(layers):
            out[l][i] = x[:, j].reshape(n_tok, -1)

    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        for i, _ in enumerate(ex.map(one, range(len(keys)))):
            if (i + 1) % 500 == 0:
                log(f'  read {model} layers {l0}-{l1 - 1}: {i + 1}/{len(keys)} ({(time.time() - t0) / (i + 1):.2f} s/ep)')
    return out


def ref_tokens(ref_root, ref, keys):
    """{layer: (N, tokens, dim) float16} for references whose stored features keep tokens."""
    if ref in POOLED_REFS:
        raise SystemExit(f'{ref}: stored features are spatially pooled; re-extract its tokens before probing')
    first = np.load(ref_root / ref / f'{keys[0]}.npz')['features']
    if ref == 'd4rt':        # (16, 16, 16, 1280): one layer, 4096 tokens
        out = {0: np.empty((len(keys), int(np.prod(first.shape[:3])), first.shape[-1]), np.float16)}
        for i, k in enumerate(keys):
            out[0][i] = np.load(ref_root / ref / f'{k}.npz')['features'].reshape(-1, first.shape[-1])
        return out
    if ref == 'dinov2':      # (33 frames, 25 layers, 1024) CLS per frame
        arr = np.stack([np.load(ref_root / ref / f'{k}.npz')['features'] for k in keys]).astype(np.float16)
        return {l: arr[:, :, l] for l in range(arr.shape[2])}
    raise ValueError(ref)


def videomae_frames(keys, workers):
    """(N, 32, 224, 224, 3) uint8: aligned GT frames 1..32, decoded once and reused across layer groups."""
    from extract_robotwin_videomaev2_features import FIRST, LAST, read_frames
    from downstream_data import gt_hdf5
    emb = keys.embodiment
    out = np.empty((len(keys), LAST - FIRST + 1, 224, 224, 3), np.uint8)

    def one(i):
        frames = read_frames({'gt_hdf5': str(gt_hdf5(emb, keys[i]))})
        out[i] = np.stack([np.asarray(f) for f in frames[FIRST:LAST + 1]])
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, range(len(keys))))
    return out


def videomae_layer_group(frames, layers, device='cuda', keep_tubelets=None):
    """keep_tubelets: optional index array over the 16 tubelets, applied before storing (halves memory for the flow probe)."""
    """{layer: (N, 3136, 1024) float16}: every token of the given VideoMAEv2 blocks, both clips concatenated."""
    model = _videomae(device)
    clip = model.config.model_config['num_frames']
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1, 1)
    out, last = {}, max(layers)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        for i in range(len(frames)):
            per_clip = {l: [] for l in layers}
            for c in range(0, frames.shape[1], clip):
                x = torch.as_tensor(frames[i, c:c + clip], device=device).permute(3, 0, 1, 2)[None].float() / 255
                x = model.model.patch_embed((x - mean) / std)
                x = model.model.pos_drop(x + model.model.pos_embed.to(x.dtype).to(device))
                for l, block in enumerate(model.model.blocks[:last + 1]):
                    x = block(x)
                    if l in per_clip:
                        per_clip[l].append(x[0])
            for l in layers:
                t = torch.cat(per_clip[l])
                if keep_tubelets is not None:
                    t = t.reshape(16, -1, t.shape[-1])[keep_tubelets].reshape(-1, t.shape[-1])
                t = t.to(torch.float16).cpu().numpy()
                if l not in out:
                    out[l] = np.empty((len(frames),) + t.shape, np.float16)
                out[l][i] = t
            if (i + 1) % 500 == 0:
                log(f'  videomaev2 layers {layers[0]}-{layers[-1]}: {i + 1}/{len(frames)}')
    return out


_VIDEOMAE = {}


def _videomae(device):
    """Load VideoMAEv2 once per process; every layer group reuses it."""
    if device not in _VIDEOMAE:
        sys.path.insert(0, str(VIDEOMAE_DIR))
        from transformers import AutoConfig, AutoModel
        cfg = AutoConfig.from_pretrained(str(VIDEOMAE_DIR), trust_remote_code=True)
        _VIDEOMAE[device] = AutoModel.from_pretrained(str(VIDEOMAE_DIR), config=cfg, trust_remote_code=True).to(device).eval()
    return _VIDEOMAE[device]


class Keys(list):
    def __init__(self, keys, embodiment):
        super().__init__(keys); self.embodiment = embodiment


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--embodiment', required=True, choices=['agilex', 'franka'])
    p.add_argument('--models', nargs='*', default=None, help='default: every OpenWAM model with all split episodes extracted')
    p.add_argument('--refs', nargs='*', default=list(TOKEN_REFS))
    p.add_argument('--layers', type=int, nargs='*', default=None, help='OpenWAM layers (default: all)')
    p.add_argument('--group', type=int, default=5, help='OpenWAM layers read per pass (RAM: ~11 GB per layer for Agilex)')
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--split', default='artifacts/downstream/split.json')
    p.add_argument('--output', type=Path, default=Path('artifacts/downstream/task_cls_layers'))
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--csv-name', default='layers.csv', help='separate file per concurrent run')
    p.add_argument('--layer-plan', type=Path, default=None, help='JSON {embodiment: {task: {model: [layers]}}}: probe only these layers per model (e.g. the top-10 of a previous seed)')
    a = p.parse_args()

    tr, te = split_keys(a.split, a.embodiment)
    planned = _plan_helper(a, 'cls')
    keys = Keys(tr + te, a.embodiment)
    names = np.array([k.split('__')[0] for k in keys])
    classes, y = np.unique(names, return_inverse=True)
    y_tr, y_te, n_tr = y[:len(tr)], y[len(tr):], len(tr)
    log(f'{a.embodiment}: {len(tr)} train / {len(te)} test, {len(classes)} tasks (chance {1 / len(classes):.3f}), {a.epochs} epochs')

    out_dir = a.output / a.embodiment
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / a.csv_name
    rows = list(csv.DictReader(open(csv_path))) if csv_path.exists() else []
    done = {(r['model'], int(r['layer'])) for r in rows}

    def record(model, kind, layer, n_tokens, acc, auroc):
        rows.append(dict(model=model, kind=kind, layer=layer, n_tokens=n_tokens, acc=acc, auroc=auroc, epochs=a.epochs))
        with open(csv_path, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        log(f'{model} layer {layer}: acc {acc:.3f} auroc {auroc:.3f}')

    for ref in a.refs:
        if ref == 'videomaev2':
            todo = [l for l in planned(ref, 24) if (ref, l) not in done]
            if todo:
                frames = videomae_frames(keys, a.workers)
                for g in range(0, len(todo), a.group):
                    xs = videomae_layer_group(frames, todo[g:g + a.group])
                    for l in todo[g:g + a.group]:
                        acc, auroc = probe_with_retry(xs[l][:n_tr], y_tr, xs[l][n_tr:], y_te, len(classes), a.seed, a.epochs)
                        record(ref, 'reference', l, xs[l].shape[1], acc, auroc)
                    del xs
                del frames
            continue
        xs = ref_tokens(Path(f'artifacts/{a.embodiment}_temporal_features_gt'), ref, keys)
        for l, x in xs.items():
            if (ref, l) in done or l not in planned(ref, len(xs)):
                continue
            acc, auroc = probe_with_retry(x[:n_tr], y_tr, x[n_tr:], y_te, len(classes), a.seed, a.epochs)
            record(ref, 'reference', l, x.shape[1], acc, auroc)
        del xs

    models = a.models
    if models is None:
        models = sorted(d.name for d in (OPENWAM_ROOT / a.embodiment).iterdir()
                        if all((d / f'{k}.npz').exists() for k in keys))
    for m in models:
        todo = [l for l in planned(m, 30) if (m, l) not in done]
        for g in range(0, len(todo), a.group):
            layers = todo[g:g + a.group]
            xs = openwam_layer_group(a.embodiment, m, keys, layers, a.workers)
            for l in layers:
                acc, auroc = probe_with_retry(xs[l][:n_tr], y_tr, xs[l][n_tr:], y_te, len(classes), a.seed, a.epochs)
                record(m, 'openwam', l, xs[l].shape[1], acc, auroc)
            del xs


if __name__ == '__main__':
    main()
