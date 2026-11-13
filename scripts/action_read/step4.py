#!/usr/bin/env python
"""Step 4 of wamu/action_read_subspace_plan.md.

--part novel (4b): for a pair (A = M1, B = M3 or M2) and a set, per layer, on identical inputs (same windows, action
    noise, sigma, action token): X = A's read coordinates in R_A (top k*_A), Y = B's in R_B (top k*_B).
    Reports canonical correlations (CCA, fit half), the fraction of B's R variance NOT linearly predictable from A's
    (ridge X -> Y on fit, residual variance on eval), and saves B's novel subspace = principal components of the fit
    residual (dims explaining 90% of it, at most 32), mapped to d-space through R_B:
    artifacts/action_read/R/novel_<B>vs<A>_<set>.pt  (evec per layer, d x r), used by probes.py --novel.
--part summary (4a, 4d): one table from the step 0-3 CSVs (+ 4b), monotone M1 -> M2 -> M3 trends marked.
"""
import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np
import torch

OUT = Path('artifacts/action_read'); FEAT = Path.home() / 'dataspace/action_read_features'


K_COMMON = 32   # the common k used for step 3 (k*' = 32 for M1 and M2 with mean ablation as the zero point)


def kstar(model):
    return K_COMMON


def part_novel(a):
    A, B = a.base, a.other
    kA, kB = kstar(A), kstar(B)
    mA = np.load(FEAT / f'{A}_{a.set}' / 'mcoord.npy', mmap_mode='r'); mB = np.load(FEAT / f'{B}_{a.set}' / 'mcoord.npy', mmap_mode='r')
    kA_, kB_ = json.loads((FEAT / f'{A}_{a.set}' / 'keys.json').read_text()), json.loads((FEAT / f'{B}_{a.set}' / 'keys.json').read_text())
    assert kA_ == kB_, 'dumps must cover identical windows'
    gt = np.load(OUT / 'gt' / f'{a.set}.npz', allow_pickle=True)
    half = {(str(t), int(e), int(s)): h for t, e, s, h in zip(gt['task'], gt['episode'], gt['start'], gt['half'])}
    hv = np.array([half[(t, int(e), int(s))] for t, e, s in kA_])
    fit, ev = np.nonzero(hv == 'fit')[0], np.nonzero(hv == 'eval')[0]
    RB = torch.load(OUT / 'R' / f'R_{B}_{a.set}.pt', map_location='cpu')
    rows, novel = [], {'evec': [], 'r': [], 'meta': dict(base=A, other=B, set=a.set, kA=kA, kB=kB)}
    for l in range(mA.shape[2]):
        X = lambda idx: np.asarray(mA[idx, :, l, :, :kA], np.float64).reshape(-1, kA)
        Y = lambda idx: np.asarray(mB[idx, :, l, :, :kB], np.float64).reshape(-1, kB)
        Xf, Yf, Xe, Ye = X(fit), Y(fit), X(ev), Y(ev)
        xm, ym = Xf.mean(0), Yf.mean(0); Xf, Yf, Xe, Ye = Xf - xm, Yf - ym, Xe - xm, Ye - ym
        # CCA via whitening
        def whit(Z):
            C = Z.T @ Z / len(Z) + 1e-6 * np.eye(Z.shape[1]); w, V = np.linalg.eigh(C); return V @ np.diag(w ** -0.5) @ V.T
        cc = np.linalg.svd(whit(Xf) @ (Xf.T @ Yf / len(Xf)) @ whit(Yf), compute_uv=False)
        W = np.linalg.solve(Xf.T @ Xf + 1e-3 * np.trace(Xf.T @ Xf) / kA * np.eye(kA), Xf.T @ Yf)
        unexpl = float(((Ye - Xe @ W) ** 2).sum() / (Ye ** 2).sum())
        Ef = Yf - Xf @ W
        w, V = np.linalg.eigh(Ef.T @ Ef); w, V = w[::-1], V[:, ::-1]
        r = int(min(32, np.searchsorted(np.cumsum(w) / w.sum(), 0.9) + 1))
        novel['evec'].append(torch.as_tensor(RB['evec'][l][:, :kB].double().numpy() @ V[:, :r], dtype=torch.float32)); novel['r'].append(r)
        rows.append(dict(base=A, other=B, set=a.set, layer=l, kA=kA, kB=kB, cca_top1=cc[0], cca_mean=float(cc.mean()), cca_top10_mean=float(cc[:10].mean()),
                         frac_unexplained_by_base=unexpl, novel_dims=r, n_fit_vectors=len(Xf), n_eval_vectors=len(Xe)))
        print(f'L{l}: CCA top {cc[0]:.3f} mean {cc.mean():.3f}; {B} R variance not predictable from {A}: {unexpl:.3f}; novel dims {r}', flush=True)
    torch.save(novel, OUT / 'R' / f'novel_{B}vs{A}_{a.set}.pt')
    p = OUT / 'results' / 'step4b' / f'step4b_novel_{B}vs{A}_{a.set}.csv'; p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', newline='') as f:
        w_ = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w_.writeheader(); w_.writerows(rows)
    print(f'wrote {p}')


def part_summary(a):
    R = OUT / 'results'
    rd = lambda p: list(csv.DictReader(open(p))) if Path(p).exists() else []
    s0 = rd(R / 'step0_baseline.csv'); ks = rd(R / 'step2_kstar.csv'); s2 = rd(R / 'step2_causal.csv')
    probes = [r for f in glob.glob(str(R / 'probes' / '*.csv')) for r in rd(f)]
    b3 = [r for f in glob.glob(str(R / 'step3b' / '*.csv')) for r in rd(f)]
    c3 = [r for f in glob.glob(str(R / 'step3c' / '*.csv')) for r in rd(f)]
    lower_better = {'depth_absrel', 'flow_epe3d_mm', 'ee_pos_mm', 'ee_rot_deg', 'grip2target_mm', 'future_tcp_mm'}
    rows = []
    def add(q, vals, note=''):
        v = [vals.get(m) for m in ('M1', 'M2', 'M3')]
        mono = ''
        if all(x is not None and np.isfinite(x) for x in v):
            mono = 'increasing' if v[0] < v[1] < v[2] else ('decreasing' if v[0] > v[1] > v[2] else '')
        rows.append(dict(quantity=q, M1=v[0], M2=v[1], M3=v[2], monotone=mono, note=note))
    for st in ('ID', 'OOD'):
        add(f'kstar_{st}', {m: max([int(r['kstar']) for r in ks if r['model'] == m and r['set'] == st and r['R_from'] == st and r['kstar']] or [np.nan])
                           for m in ('M1', 'M2', 'M3')}, 'max over action sigma; NaN = not reached at some sigma')
        for sg in sorted({r['action_sigma'] for r in s0}):
            add(f'gain_{st}_sigma{float(sg):.2f}', {r['model']: float(r['gain']) for r in s0 if r['set'] == st and r['action_sigma'] == sg})
            add(f'nmse_full_{st}_sigma{float(sg):.2f}', {r['model']: float(r['nmse_full']) for r in s0 if r['set'] == st and r['action_sigma'] == sg})
    # retention per content and layer (same-set probes): (score_R - mean rand) / (score_full - mean rand)
    for st in ('ID', 'OOD'):
        P = [r for r in probes if r['train_set'] == st and r['test_set'] == st]
        for metric in sorted({r['metric'] for r in P}):
            for layer in sorted({int(r['layer']) for r in P}):
                vals = {}
                for m in ('M1', 'M2', 'M3'):
                    g = lambda inp: [float(r['value']) for r in P if r['model'] == m and r['metric'] == metric and int(r['layer']) == layer and r['input'] == inp]
                    full, rr = g('full'), g('R'); rnd = [x for s in range(5) for x in g(f'rand{s}')]
                    if full and rr and rnd:
                        den = np.mean(full) - np.mean(rnd)
                        vals[m] = (np.mean(rr) - np.mean(rnd)) / den if abs(den) > 1e-12 else np.nan
                add(f'retention_{st}_{metric}_L{layer}', vals, 'lower-is-better metric' if metric in lower_better else '')
    # 4d: ID->OOD probe degradation on the full input and on R
    for metric in sorted({r['metric'] for r in probes}):
        for inp in ('full', 'R'):
            vals = {}
            for m in ('M1', 'M2', 'M3'):
                g = lambda tr, te: [float(r['value']) for r in probes if r['model'] == m and r['metric'] == metric and r['input'] == inp and r['train_set'] == tr and r['test_set'] == te]
                a_, b_ = g('ID', 'ID'), g('ID', 'OOD')
                if a_ and b_:
                    vals[m] = (np.mean(b_) - np.mean(a_)) / abs(np.mean(a_))
            add(f'ID2OOD_rel_change_{metric}_{inp}', vals, 'relative change of the score, all dumped layers pooled')
    for sg in sorted({r['action_sigma'] for r in s2}):
        for rf in ('ID', 'OOD'):
            vals = {}
            for m in ('M1', 'M2', 'M3'):
                k = [int(r['kstar']) for r in ks if r['model'] == m and r['set'] == 'ID' and r['R_from'] == 'ID' and r['kstar']]
                if not k:
                    continue
                kk = min([int(r['k']) for r in s2], key=lambda x: abs(x - max(k)))
                v = [float(r['rec_keepR']) for r in s2 if r['model'] == m and r['set'] == 'OOD' and r['R_from'] == rf and r['action_sigma'] == sg and int(r['k']) == kk]
                if v:
                    vals[m] = v[0]
            add(f'rec_keepR_OOD_Rfrom{rf}_sigma{float(sg):.2f}_at_kstar', vals)
    for st in ('ID', 'OOD'):
        add(f'unexplained_3b_{st}_mean_over_layers', {m: float(np.mean([float(r['unexplained']) for r in b3 if r['model'] == m and r['set'] == st])) for m in ('M1', 'M2', 'M3')
                                                     if any(r['model'] == m and r['set'] == st for r in b3)})
        C = [r for r in c3 if r['set'] == st and r['latent_frame'] == 'all' and abs(float(r['action_sigma']) - 0.5) < 0.01]
        for reg in sorted({r['region'] for r in C}):
            add(f'enrichment_{st}_{reg}_mean_over_layers_sigma0.50', {m: float(np.mean([float(r['enrichment']) for r in C if r['model'] == m and r['region'] == reg]))
                                                                      for m in ('M1', 'M2', 'M3') if any(r['model'] == m and r['region'] == reg for r in C)})
    for f in sorted(glob.glob(str(R / 'step4b' / '*.csv'))):
        B4 = rd(f)
        if B4:
            add(f'novel_frac_unexplained_{B4[0]["other"]}vs{B4[0]["base"]}_{B4[0]["set"]}_mean_over_layers',
                {B4[0]['other']: float(np.mean([float(r['frac_unexplained_by_base']) for r in B4]))})
    p = R / 'step4_summary.csv'
    with p.open('w', newline='') as f:
        w_ = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w_.writeheader(); w_.writerows(rows)
    print(f'wrote {p} ({len(rows)} rows; {sum(1 for r in rows if r["monotone"])} monotone)')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--part', choices=['novel', 'summary'], required=True)
    p.add_argument('--base', default='M1'); p.add_argument('--other', default='M3'); p.add_argument('--set', default='ID')
    a = p.parse_args()
    {'novel': part_novel, 'summary': part_summary}[a.part](a)


if __name__ == '__main__':
    main()
