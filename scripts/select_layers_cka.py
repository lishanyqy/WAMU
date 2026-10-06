#!/usr/bin/env python
"""Pick one OpenWAM layer per downstream task by CKA against that task's reference model.

Samples are (episode, latent) pairs from the probe *train* split only.  Every reference is
aggregated onto OpenWAM's causal-VAE latents (latent 0 = aligned frame 0, latent k = 4k-3..4k):
  dinov2     (33 frames, 25 layers, 1024)  -> mean of the frames in each latent      latents 0..8
  depthv3    (33 frames, 16, 16)           -> mean depth map per latent, flattened   latents 0..8
  videomaev2 (16 tubelets, 24 layers, 1024)-> mean of the 2 tubelets in each latent  latents 1..8
  d4rt       (16, 16, 16, 1280)            -> spatial mean, then 2 tubelets/latent   latents 1..8
OpenWAM (9, 30, 8, 10, 3072) head tokens are mean-pooled over space; pooled train features are
cached so later runs and other references skip the 133 MB/episode reads.

For each OpenWAM layer the score is the mean (and max) CKA over the reference's layers; the
selected layer maximises the mean, following wamu/downstreamtask.md.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

TASKS = {  # downstream task -> (reference, CKA kind), from wamu/downstreamtask.md
    'depth': ('depthv3', 'linear'),
    'segmentation': ('dinov2', 'linear'),
    'd4rt': ('d4rt', 'rbf'),
    'videomae': ('videomaev2', 'linear'),
}
N_LATENTS = 9


def latent_of_aligned(frame):
    return 0 if frame == 0 else (frame - 1) // 4 + 1


def ref_latents(name, path):
    """Return {latent: (ref_layers, dim)} for one episode."""
    z = np.load(path)
    f = z['features'].astype(np.float32)
    if name in ('dinov2', 'depthv3'):
        if name == 'depthv3':
            f = f.reshape(f.shape[0], 1, -1)
        groups = {}
        for frame in range(f.shape[0]):
            groups.setdefault(latent_of_aligned(frame), []).append(f[frame])
    else:
        if name == 'd4rt':
            f = f.mean(axis=(1, 2))[:, None, :]
        groups = {}
        for j, frames in enumerate(z['tubelet_aligned_frames']):
            latents = {latent_of_aligned(int(x)) for x in frames}
            assert len(latents) == 1, (path, frames)
            groups.setdefault(latents.pop(), []).append(f[j])
    return {k: np.mean(v, axis=0) for k, v in groups.items()}


def openwam_pooled(model_dir, keys, cache):
    if cache.exists():
        z = np.load(cache)
        if list(z['keys']) == keys:
            return z['features']
    out = np.zeros((len(keys), N_LATENTS, 30, 3072), np.float32)
    for i, k in enumerate(keys):
        f = np.load(model_dir / f'{k}.npz')['features']  # (9, 30, 8, 10, 3072)
        out[i] = f.astype(np.float32).mean(axis=(2, 3))
        if (i + 1) % 50 == 0:
            print(f'  pooled {i + 1}/{len(keys)}', flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, features=out.astype(np.float16), keys=np.array(keys))
    return out


def centered_gram(x):
    k = x @ x.T
    return k - k.mean(0, keepdim=True) - k.mean(1, keepdim=True) + k.mean()


def rbf_centered_gram(x):
    # Same protocol as scripts/score_robotwin_rbf_cka.py: standardise, median-heuristic bandwidth.
    x = (x - x.mean(0, keepdim=True)) / x.std(0, keepdim=True).clamp_min(1e-6)
    sq = (x * x).sum(1, keepdim=True)
    d = (sq + sq.T - 2 * x @ x.T).clamp_min(0)
    iu = torch.triu_indices(len(x), len(x), 1, device=x.device)
    gamma = 1.0 / (2.0 * d[iu[0], iu[1]].median()).clamp_min(1e-6)
    return centered_gram_from_kernel(torch.exp(-gamma * d))


def centered_gram_from_kernel(k):
    return k - k.mean(0, keepdim=True) - k.mean(1, keepdim=True) + k.mean()


def cka(kx, ky):
    return float((kx * ky).sum() / ((kx * kx).sum().sqrt() * (ky * ky).sum().sqrt()).clamp_min(1e-12))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--embodiment', required=True, choices=['agilex', 'franka'])
    p.add_argument('--split', type=Path, default=Path('artifacts/downstream/split.json'))
    p.add_argument('--openwam-root', type=Path, default=Path('/home/a1905721/dataspace/openwam_gt_features'))
    p.add_argument('--ref-root', type=Path, default=None, help='default: artifacts/<embodiment>_temporal_features_gt')
    p.add_argument('--models', nargs='*', default=None, help='default: every model dir with all train episodes')
    p.add_argument('--output', type=Path, default=Path('artifacts/downstream/layer_selection'))
    p.add_argument('--rbf-max-points', type=int, default=3000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    ref_root = a.ref_root or Path(f'artifacts/{a.embodiment}_temporal_features_gt')
    keys = json.loads(a.split.read_text())[a.embodiment]['train']
    out_dir = a.output / a.embodiment
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)

    model_root = a.openwam_root / a.embodiment
    models = a.models or sorted(d.name for d in model_root.iterdir() if d.is_dir())
    ready = [m for m in models if all((model_root / m / f'{k}.npz').exists() for k in keys)]
    for m in sorted(set(models) - set(ready)):
        print(f'skip {m}: train split not fully extracted yet', flush=True)

    # Reference samples, one gram per reference layer, shared across OpenWAM models.
    refs = {}
    for task, (ref, kind) in TASKS.items():
        if ref in refs:
            continue
        per_ep = [ref_latents(ref, ref_root / ref / f'{k}.npz') for k in keys]
        latents = sorted(set.intersection(*(set(e) for e in per_ep)))
        y = np.stack([e[l] for e in per_ep for l in latents])  # (samples, ref_layers, dim)
        refs[ref] = {'latents': latents, 'y': y}
        print(f'{ref}: samples={len(y)} latents={latents} layers={y.shape[1]} dim={y.shape[2]}', flush=True)

    rng = np.random.default_rng(a.seed)
    rows = []
    for m in ready:
        print(f'== {m}', flush=True)
        x_all = openwam_pooled(model_root / m, keys, out_dir / 'cache' / f'{m}.npz')  # (eps, 9, 30, 3072)
        for task, (ref, kind) in TASKS.items():
            r = refs[ref]
            x = x_all[:, r['latents']].reshape(-1, x_all.shape[2], x_all.shape[3])  # (samples, 30, 3072)
            y = r['y']
            idx = np.arange(len(x))
            if kind == 'rbf' and len(idx) > a.rbf_max_points:
                idx = np.sort(rng.choice(len(idx), a.rbf_max_points, replace=False))
            gram = rbf_centered_gram if kind == 'rbf' else lambda t: centered_gram(t - t.mean(0, keepdim=True))
            to = lambda arr: torch.as_tensor(arr[idx], device=dev, dtype=torch.float32)
            ky = [gram(to(y[:, l])) for l in range(y.shape[1])]
            for vl in range(x.shape[1]):
                kx = gram(to(x[:, vl]))
                scores = [cka(kx, k) for k in ky]
                rows.append({'embodiment': a.embodiment, 'model': m, 'task': task, 'reference': ref, 'cka': kind,
                             'openwam_layer': vl, 'mean_over_ref_layers': float(np.mean(scores)),
                             'max_over_ref_layers': float(np.max(scores)), 'best_ref_layer': int(np.argmax(scores)),
                             'samples': len(idx)})
            del ky
            best = max((r_ for r_ in rows if r_['model'] == m and r_['task'] == task), key=lambda r_: r_['mean_over_ref_layers'])
            print(f'  {task:12s} {ref:10s} {kind:6s} n={len(idx)} -> layer {best["openwam_layer"]} '
                  f'(mean {best["mean_over_ref_layers"]:.4f}, max {best["max_over_ref_layers"]:.4f})', flush=True)
        torch.cuda.empty_cache()

    if not rows:
        raise SystemExit('no model has the full train split extracted yet')
    with (out_dir / 'all_layers.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    selected = {}
    for m in ready:
        selected[m] = {}
        for task in TASKS:
            z = [r_ for r_ in rows if r_['model'] == m and r_['task'] == task]
            by_mean = max(z, key=lambda r_: r_['mean_over_ref_layers'])
            by_max = max(z, key=lambda r_: r_['max_over_ref_layers'])
            selected[m][task] = {'layer': by_mean['openwam_layer'], 'mean_cka': by_mean['mean_over_ref_layers'],
                                 'layer_by_max': by_max['openwam_layer'], 'max_cka': by_max['max_over_ref_layers'],
                                 'best_ref_layer_at_max': by_max['best_ref_layer']}
    (out_dir / 'selected_layers.json').write_text(json.dumps(
        {'embodiment': a.embodiment, 'split': str(a.split), 'train_episodes': len(keys), 'tasks': TASKS,
         'rule': 'argmax over OpenWAM layers of mean CKA across reference layers', 'selected': selected}, indent=1))
    print(json.dumps(selected, indent=1))


if __name__ == '__main__':
    main()
