#!/usr/bin/env python
"""Steps 0 and 2 tables/figures of wamu/action_read_subspace_plan.md from read_subspace.py --mode causal outputs.

Readouts per (model, set, action sigma), means over EVAL windows, 95% CI from 1000 bootstrap resamples of episodes:
  NMSE_full, NMSE_cut, gain = NMSE_cut / NMSE_full
  rec(X) = (NMSE_cut - NMSE_X) / (NMSE_cut - NMSE_full)   for keep/remove-R(k), random(k) (mean +- std over seeds),
  5-layer windows (keep/remove R(k), zero).
Acceptance (step 2): k* = smallest k with rec(keep-R) >= 0.9, rec(remove-R) <= 0.1, keep-R above EVERY random seed's
keep and remove-R below EVERY random seed's remove at the same k.  Evaluated per sigma; a model's read dimension is
reported only if k* exists at every sigma (max over sigmas); otherwise 'not reached' (no thresholds are tuned).

usage: python scripts/action_read/analyze_causal.py [--rerun-a RAW --rerun-b RAW]
"""
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

RAW = Path('artifacts/action_read/raw'); RES = Path('artifacts/action_read/results'); FIG = Path('artifacts/action_read/figs')
N_BOOT, BOOT_SEED = 1000, 0


def load(path):
    z = np.load(path, allow_pickle=True)
    return dict(meta=json.loads(str(z['meta'])), variants=[str(v) for v in z['variants']], key=z['key'], sigma=z['action_sigma'], nmse=z['nmse'])


def ep_sums(d, sig):
    sel = np.isclose(d['sigma'], sig, atol=1e-3)
    keys = [f'{k[0]}/{k[1]}' for k in d['key'][sel]]
    eps = sorted(set(keys)); pos = {e: i for i, e in enumerate(eps)}
    S = np.zeros((len(eps), d['nmse'].shape[1])); n = np.zeros(len(eps))
    for kk, row in zip(keys, d['nmse'][sel]):
        S[pos[kk]] += row; n[pos[kk]] += 1
    return S, n, eps


def boot_means(S, n):
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(n), size=(N_BOOT, len(n)))
    return S.sum(0) / n.sum(), S[idx].sum(1) / n[idx].sum(1)[:, None]          # (V,), (N_BOOT, V)


def rec(means, vi, full, cut):
    return (means[..., cut] - means[..., vi]) / (means[..., cut] - means[..., full])


def ci(x):
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def analyze(d, rows2, rows0, rowsw, curves):
    m = d['meta']; V = d['variants']; vi = {v: i for i, v in enumerate(V)}
    full, cut = vi['full'], vi['cut']
    ks = sorted({int(x.split('_')[1]) for x in V if re.fullmatch(r'keepR_\d+', x)})
    seeds = sorted({int(re.match(r'keepRand(\d+)_', x).group(1)) for x in V if x.startswith('keepRand')})
    rfrom = m.get('r_from', m['set'])
    for sig in sorted(set(np.round(d['sigma'], 4))):
        S, n, eps = ep_sums(d, sig)
        pm, bm = boot_means(S, n)
        rows0.append(dict(model=m['model'], set=m['set'], action_sigma=sig, n_episodes=len(eps), n_windows=int(n.sum()),
                          nmse_full=pm[full], nmse_full_lo=ci(bm[:, full])[0], nmse_full_hi=ci(bm[:, full])[1],
                          nmse_cut=pm[cut], nmse_cut_lo=ci(bm[:, cut])[0], nmse_cut_hi=ci(bm[:, cut])[1],
                          gain=pm[cut] / pm[full], gain_lo=ci(bm[:, cut] / bm[:, full])[0], gain_hi=ci(bm[:, cut] / bm[:, full])[1],
                          seed=m['seed'], boot_seed=BOOT_SEED, n_boot=N_BOOT, replay_max_rel_diff=m['replay_max_rel_diff']))
        kstar = None; cur = dict(k=ks, keep=[], remove=[], keep_rand=[], remove_rand=[], keep_rand_sd=[], remove_rand_sd=[])
        for k in ks:
            rk, rr = rec(pm, vi[f'keepR_{k}'], full, cut), rec(pm, vi[f'removeR_{k}'], full, cut)
            bk, br = rec(bm, vi[f'keepR_{k}'], full, cut), rec(bm, vi[f'removeR_{k}'], full, cut)
            rkr = np.array([rec(pm, vi[f'keepRand{s}_{k}'], full, cut) for s in seeds]) if seeds else np.array([np.nan])
            rrr = np.array([rec(pm, vi[f'removeRand{s}_{k}'], full, cut) for s in seeds]) if seeds else np.array([np.nan])
            ok = rk >= 0.9 and rr <= 0.1 and (rk > rkr.max()) and (rr < rrr.min())
            if ok and kstar is None:
                kstar = k
            rows2.append(dict(model=m['model'], set=m['set'], R_from=rfrom, action_sigma=sig, k=k, rec_keepR=rk, rec_keepR_lo=ci(bk)[0], rec_keepR_hi=ci(bk)[1],
                              rec_removeR=rr, rec_removeR_lo=ci(br)[0], rec_removeR_hi=ci(br)[1],
                              rec_keepRand_mean=float(rkr.mean()), rec_keepRand_std=float(rkr.std()), rec_removeRand_mean=float(rrr.mean()),
                              rec_removeRand_std=float(rrr.std()), n_random_seeds=len(seeds), meets_criterion=ok, seed=m['seed']))
            cur['keep'].append(rk); cur['remove'].append(rr); cur['keep_rand'].append(rkr.mean()); cur['remove_rand'].append(rrr.mean())
            cur['keep_rand_sd'].append(rkr.std()); cur['remove_rand_sd'].append(rrr.std())
        cur['kstar'] = kstar
        curves[(m['model'], m['set'], rfrom, sig)] = cur
        for v in V:
            mm = re.fullmatch(r'win(\d+)_(keepR_\d+|removeR_\d+|zero)', v)
            if mm:
                b = rec(bm, vi[v], full, cut)
                rowsw.append(dict(model=m['model'], set=m['set'], R_from=rfrom, action_sigma=sig, layers=f'{int(mm.group(1))}-{int(mm.group(1)) + 4}',
                                  intervention=mm.group(2), rec=rec(pm, vi[v], full, cut), rec_lo=ci(b)[0], rec_hi=ci(b)[1], seed=m['seed']))


def write(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f'wrote {path} ({len(rows)} rows)')


def plot(curves):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    FIG.mkdir(parents=True, exist_ok=True)
    groups = sorted({(k[1], k[2]) for k in curves})
    with PdfPages(FIG / 'step2_recovery_curves.pdf') as pdf:
        for st, rf in groups:
            sigs = sorted({k[3] for k in curves if k[1] == st and k[2] == rf}); models = sorted({k[0] for k in curves if k[1] == st and k[2] == rf})
            fig, axes = plt.subplots(len(models), len(sigs), figsize=(4 * len(sigs), 3.2 * len(models)), squeeze=False)
            for i, mo in enumerate(models):
                for j, sg in enumerate(sigs):
                    ax = axes[i, j]; c = curves.get((mo, st, rf, sg))
                    if c is None:
                        ax.axis('off'); continue
                    k = np.array(c['k'])
                    ax.plot(k, c['keep'], 'o-', color='#1f6fb4', label='keep-R(k)'); ax.plot(k, c['remove'], 's-', color='#c0392b', label='remove-R(k)')
                    for key, col in (('keep_rand', '#1f6fb4'), ('remove_rand', '#c0392b')):
                        mu, sd = np.array(c[key]), np.array(c[key + '_sd'])
                        ax.plot(k, mu, '--', color=col, alpha=.6, label=key.replace('_rand', ' random(k)')); ax.fill_between(k, mu - sd, mu + sd, color=col, alpha=.12)
                    ax.axhline(0.9, color='grey', lw=.6, ls=':'); ax.axhline(0.1, color='grey', lw=.6, ls=':')
                    ax.set_xscale('log', base=2); ax.set_ylim(-0.3, 1.15)
                    ax.set_title(f'{mo} {st} (R from {rf}) action sigma {sg:.2f}' + (f'  k*={c["kstar"]}' if c['kstar'] else '  k* not reached'), fontsize=8)
                    ax.set_xlabel('k'); ax.set_ylabel('rec')
                    if i == 0 and j == 0:
                        ax.legend(fontsize=6)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    print(f'wrote {FIG / "step2_recovery_curves.pdf"}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rerun-a', type=Path, default=None); p.add_argument('--rerun-b', type=Path, default=None)
    a = p.parse_args()
    rows2, rows0, rowsw, curves = [], [], [], {}
    for f in sorted(RAW.glob('causal_M*_*.npz')):
        if re.search(r'_(smoke|rerun|mean)', f.stem):
            continue
        d = load(f)
        rfrom = re.search(r'_R(ID|OOD)', f.stem).group(1); d['meta']['r_from'] = rfrom
        rows2_, rows0_ = [], []
        analyze(d, rows2_ if True else None, rows0_, rowsw, curves)
        rows2 += rows2_
        if rfrom == d['meta']['set']:
            rows0 += rows0_
    write(RES / 'step0_baseline.csv', rows0); write(RES / 'step2_causal.csv', rows2); write(RES / 'step2_layer_windows.csv', rowsw)
    summ = []
    for (mo, st, rf, sg), c in sorted(curves.items()):
        summ.append(dict(model=mo, set=st, R_from=rf, action_sigma=sg, kstar=c['kstar'] if c['kstar'] else ''))
    write(RES / 'step2_kstar.csv', summ)
    for r in summ:
        print(r)
    if curves:
        plot(curves)
    if a.rerun_a and a.rerun_b:
        A, B = load(a.rerun_a), load(a.rerun_b)
        same_keys = np.array_equal(A['key'], B['key']) and np.array_equal(A['sigma'], B['sigma'])
        same = same_keys and np.array_equal(A['nmse'], B['nmse'])
        diff = float(np.abs(A['nmse'] - B['nmse']).max()) if A['nmse'].shape == B['nmse'].shape else float('nan')
        print(f'rerun check: windows identical {same_keys}; all readouts bit-identical {same}; max |diff| {diff:.3g}')
        (RES / 'step0_rerun_check.json').write_text(json.dumps(dict(a=str(a.rerun_a), b=str(a.rerun_b), windows_identical=bool(same_keys),
                                                                     bit_identical=bool(same), max_abs_diff=diff), indent=1))


if __name__ == '__main__':
    main()
