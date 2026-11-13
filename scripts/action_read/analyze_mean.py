#!/usr/bin/env python
"""Mean-ablation control for step 2 (wamu/action_read_subspace_plan.md, follow-up).

mean-ablate: on every layer the action rows' read vector is replaced by its layer mean, m' = mu (= remove-R at full
rank); attention weights and the video stream are untouched.  Run by read_subspace.py --mode causal --mean-only with the
same windows and noise as the main run (checked here: full and cut must reproduce the saved values).
Reported per (model, set, action sigma), means over EVAL windows, 95% CI from 1000 episode-bootstrap resamples:
  NMSE_mean;  rec(mean) = (NMSE_cut - NMSE_mean) / (NMSE_cut - NMSE_full)
  rec'(X) = (NMSE_mean - NMSE_X) / (NMSE_mean - NMSE_full) for every saved keep/remove-R(k) and random(k) readout
  k*' = smallest k with rec'(keep-R) >= 0.9 and rec'(remove-R) <= 0.1  (beats-random reported as a separate column);
  the original-definition k* is listed next to it.
outputs: results/step2_mean_baseline.csv, results/step2_causal_meanref.csv, results/step2_kstar_meanref.csv
"""
import csv
import json
import re
from pathlib import Path

import numpy as np

RAW = Path('artifacts/action_read/raw'); RES = Path('artifacts/action_read/results')
N_BOOT = 1000


def load(p):
    z = np.load(p, allow_pickle=True)
    return dict(variants=[str(v) for v in z['variants']], key=[tuple(map(str, k)) for k in z['key']], sigma=np.asarray(z['action_sigma'], float), nmse=np.asarray(z['nmse'], float))


def write(p, rows):
    if rows:
        with p.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(f'wrote {p} ({len(rows)} rows)')


def main():
    orig_k = {(r['model'], r['set'], r['R_from'], round(float(r['action_sigma']), 2)): r['kstar'] for r in csv.DictReader(open(RES / 'step2_kstar.csv'))} \
        if (RES / 'step2_kstar.csv').exists() else {}
    base_rows, curve_rows, k_rows = [], [], []
    for mf in sorted(RAW.glob('causal_M*_*_R*_mean.npz')):
        model, set_, rf = re.match(r'causal_(M\d)_(ID|OOD)_R(ID|OOD)_mean', mf.stem).groups()
        main_f = RAW / f'causal_{model}_{set_}_R{rf}.npz'
        if not main_f.exists():
            print(f'skip {mf.name}: no main run'); continue
        A, B = load(main_f), load(mf)
        assert A['key'] == B['key'] and np.array_equal(A['sigma'], B['sigma']), f'{mf.name}: windows / sigmas differ from the main run'
        va, vb = {v: i for i, v in enumerate(A['variants'])}, {v: i for i, v in enumerate(B['variants'])}
        dfull = np.abs(A['nmse'][:, va['full']] - B['nmse'][:, vb['full']]).max(); dcut = np.abs(A['nmse'][:, va['cut']] - B['nmse'][:, vb['cut']]).max()
        rel = dfull / max(np.abs(A['nmse'][:, va['full']]).max(), 1e-12)
        print(f'{model} {set_} R{rf}: alignment check  max|full diff| {dfull:.3g} (rel {rel:.3g}), max|cut diff| {dcut:.3g}')
        X = np.concatenate([A['nmse'], B['nmse'][:, [vb['mean']]]], 1); V = A['variants'] + ['mean']; vi = {v: i for i, v in enumerate(V)}
        ks = sorted({int(v.split('_')[1]) for v in V if re.fullmatch(r'keepR_\d+', v)})
        seeds = sorted({int(re.match(r'keepRand(\d+)_', v).group(1)) for v in V if v.startswith('keepRand')})
        for sig in sorted({round(s, 4) for s in A['sigma']}):
            sel = np.isclose(A['sigma'], sig, atol=1e-3)
            eps = [f'{k[0]}/{k[1]}' for k, s_ in zip(A['key'], sel) if s_]
            ue = sorted(set(eps)); pos = {e: i for i, e in enumerate(ue)}
            S = np.zeros((len(ue), X.shape[1])); n = np.zeros(len(ue))
            for e, row in zip(eps, X[sel]):
                S[pos[e]] += row; n[pos[e]] += 1
            pm = S.sum(0) / n.sum()
            bi = np.random.default_rng(0).integers(0, len(ue), size=(N_BOOT, len(ue))); bm = S[bi].sum(1) / n[bi].sum(1)[:, None]
            ci = lambda x: (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))
            full, cut, mean = vi['full'], vi['cut'], vi['mean']
            rec = lambda M, j: (M[..., cut] - M[..., j]) / (M[..., cut] - M[..., full])
            recp = lambda M, j: (M[..., mean] - M[..., j]) / (M[..., mean] - M[..., full])
            base_rows.append(dict(model=model, set=set_, R_from=rf, action_sigma=sig, n_episodes=len(ue), n_windows=int(n.sum()),
                                  nmse_full=pm[full], nmse_cut=pm[cut], nmse_mean=pm[mean], nmse_mean_lo=ci(bm[:, mean])[0], nmse_mean_hi=ci(bm[:, mean])[1],
                                  rec_mean=rec(pm, mean), rec_mean_lo=ci(rec(bm, mean))[0], rec_mean_hi=ci(rec(bm, mean))[1],
                                  mean_over_cut=pm[mean] / pm[cut], full_alignment_max_abs_diff=float(dfull), cut_alignment_max_abs_diff=float(dcut)))
            kstar = None
            for k in ks:
                kk, rr = vi[f'keepR_{k}'], vi[f'removeR_{k}']
                rk, rm = recp(pm, kk), recp(pm, rr)
                rnd_k = np.array([recp(pm, vi[f'keepRand{s}_{k}']) for s in seeds]); rnd_r = np.array([recp(pm, vi[f'removeRand{s}_{k}']) for s in seeds])
                meets = bool(rk >= 0.9 and rm <= 0.1); beats = bool(rk > rnd_k.max() and rm < rnd_r.min()) if seeds else False
                if meets and kstar is None:
                    kstar = k; kstar_beats = beats
                curve_rows.append(dict(model=model, set=set_, R_from=rf, action_sigma=sig, k=k,
                                       recp_keepR=rk, recp_keepR_lo=ci(recp(bm, kk))[0], recp_keepR_hi=ci(recp(bm, kk))[1],
                                       recp_removeR=rm, recp_removeR_lo=ci(recp(bm, rr))[0], recp_removeR_hi=ci(recp(bm, rr))[1],
                                       recp_keepRand_mean=float(rnd_k.mean()), recp_keepRand_std=float(rnd_k.std()),
                                       recp_removeRand_mean=float(rnd_r.mean()), recp_removeRand_std=float(rnd_r.std()),
                                       rec_keepR=rec(pm, kk), rec_removeR=rec(pm, rr), meets_criterion_meanref=meets, beats_all_random=beats))
            k_rows.append(dict(model=model, set=set_, R_from=rf, action_sigma=sig, kstar_meanref=kstar or '',
                               kstar_meanref_beats_random=(kstar_beats if kstar else ''), kstar_original=orig_k.get((model, set_, rf, round(sig, 2)), '')))
            print(f'  sigma {sig:.2f}: NMSE full {pm[full]:.5f} cut {pm[cut]:.4f} mean {pm[mean]:.4f}  rec(mean) {rec(pm, mean):.3f}  k*\' {kstar}  (original k* {orig_k.get((model, set_, rf, round(sig, 2)), "")})')
    write(RES / 'step2_mean_baseline.csv', base_rows); write(RES / 'step2_causal_meanref.csv', curve_rows); write(RES / 'step2_kstar_meanref.csv', k_rows)


if __name__ == '__main__':
    main()
