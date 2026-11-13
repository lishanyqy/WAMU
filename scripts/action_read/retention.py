#!/usr/bin/env python
"""Step 3a retention table from probes.py outputs (wamu/action_read_subspace_plan.md).

retention = (score_R - mean score_rand) / (score_full - mean score_rand)   (error metrics: the same formula works,
both numerator and denominator flip sign).  Scores of window-level probes are first averaged over probe seeds.
Also reported: score_R - mean rand in units of the std over the 5 random subspaces (z), so 'R keeps this content better
than a random subspace of the same rank' can be read off directly.
Output: results/step3a_retention.csv (+ printed summary over the main read layers 14 and 19 and all dumped layers).
"""
import csv
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np

RES = Path('artifacts/action_read/results')
LOWER = {'depth_absrel', 'flow_epe3d_mm', 'ee_pos_mm', 'ee_rot_deg', 'grip2target_mm', 'future_tcp_mm'}
NICE = {'depth_absrel': 'depth (AbsRel)', 'seg_iou': 'target seg (IoU)', 'flow_epe3d_mm': '3D flow (EPE mm)', 'track_aj': '2D track (AJ)',
        'task_acc': 'task (Acc)', 'ee_pos_mm': 'EE position (mm)', 'ee_rot_deg': 'EE rotation (deg)', 'grip2target_mm': 'gripper->target (mm)',
        'contact_acc': 'contact/grasp (Acc)', 'future_tcp_mm': 'future EE traj (mm)'}


def main():
    rows = [r for f in sorted(glob.glob(str(RES / 'probes' / '*.csv'))) for r in csv.DictReader(open(f))]
    S = defaultdict(list)
    for r in rows:
        S[(r['model'], r['train_set'], r['test_set'], int(r['layer']), r['metric'], r['input'])].append(float(r['value']))
    out = []
    keys = sorted({k[:5] for k in S})
    for (m, tr, te, l, met) in keys:
        g = lambda inp: np.mean(S[(m, tr, te, l, met, inp)]) if (m, tr, te, l, met, inp) in S else np.nan
        full, rr = g('full'), g('R'); rnd = np.array([g(f'rand{s}') for s in range(5)]); rnd = rnd[np.isfinite(rnd)]
        if not len(rnd) or not np.isfinite(full) or not np.isfinite(rr):
            continue
        den = full - rnd.mean()
        ret = (rr - rnd.mean()) / den if abs(den) > 1e-9 else np.nan
        sd = rnd.std(ddof=1) if len(rnd) > 1 else np.nan
        better = (rnd.mean() - rr) if met in LOWER else (rr - rnd.mean())
        out.append(dict(model=m, train_set=tr, test_set=te, layer=l, metric=met, lower_is_better=met in LOWER, score_full=full, score_R=rr,
                        score_rand_mean=rnd.mean(), score_rand_std=sd, n_rand=len(rnd), retention=ret, R_minus_rand_in_rand_std=better / sd if sd and sd > 0 else np.nan,
                        n_probe_seeds=len(S[(m, tr, te, l, met, 'R')])))
    p = RES / 'step3a_retention.csv'
    with p.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    print(f'wrote {p} ({len(out)} rows)')
    for (m, tr, te) in sorted({(r['model'], r['train_set'], r['test_set']) for r in out}):
        sub = [r for r in out if (r['model'], r['train_set'], r['test_set']) == (m, tr, te)]
        layers = sorted({r['layer'] for r in sub})
        print(f'\n{m} train {tr} -> test {te}: retention by layer ' + ' '.join(f'L{l:>2d}' for l in layers) + '  | full / R / rand at L14')
        for met in NICE:
            rr = {r['layer']: r for r in sub if r['metric'] == met}
            if not rr:
                continue
            r14 = rr.get(14)
            tail = f'  | {r14["score_full"]:.3g} / {r14["score_R"]:.3g} / {r14["score_rand_mean"]:.3g}±{r14["score_rand_std"]:.2g} (z {r14["R_minus_rand_in_rand_std"]:+.1f})' if r14 else ''
            print(f'  {NICE[met]:22s} ' + ' '.join(f'{rr[l]["retention"]:+.2f}' if l in rr else '  -  ' for l in layers) + tail)


if __name__ == '__main__':
    main()
