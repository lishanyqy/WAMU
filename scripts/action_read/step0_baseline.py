#!/usr/bin/env python
"""Step 0 of wamu/action_read_subspace_plan.md: NMSE_full / NMSE_cut / gain per model, data set and action sigma, on the
EVAL half of artifacts/action_read/split.json, from attention_video_share.py outputs (no new forwards).

Per sample, attention_video_share.py stores nmse = sum(e^2)/sum(target^2) with video and nmse_abl = the same with the
action rows cut off from every video key. Reported: mean over eval windows; gain = mean(nmse_cut) / mean(nmse_full);
95% CI from 1000 bootstrap resamples of eval EPISODES (task, episode), seed fixed.

usage: python scripts/action_read/step0_baseline.py [--rerun artifacts/action_read/rerun_M1_ID.npz]
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

MODELS = {'M1': 'sft_from_scratch_robotwin_clean', 'M2': 'sft_robot_only_robotwin_clean',
          'M3': 'sft_ego_robot_cotrain_robotwin_clean_action_sees_video'}
SETS = {'ID': ('aloha-c2c', 'clean_50'), 'OOD': ('aloha-c2r', 'randomized_500')}
FARM = Path.home() / 'dataspace/openwam-attn-aligned'
N_BOOT, BOOT_SEED = 1000, 0


def ds_index_to_episode(task, variant):
    """RoboTwinDataset sorts the episode paths as strings (episode0, episode1, episode10, ...)."""
    files = sorted(str(p) for p in (FARM / task / f'aloha-agilex_{variant}' / 'data').glob('episode*.hdf5'))
    return [int(Path(f).stem[len('episode'):]) for f in files]


def load_rows(npz_path, variant, eval_eps):
    z = np.load(npz_path, allow_pickle=True)
    cols = list(z['abl_cols']); abl = z['abl']; meta = z['abl_meta']
    maps = {}
    rows = []
    for (task, ep_idx, start), r in zip(meta, abl):
        if task not in maps:
            maps[task] = ds_index_to_episode(task, variant)
        ep = maps[task][int(ep_idx)]
        if ep in eval_eps[task]:
            rows.append((task, ep, int(start), r[cols.index('action_sigma')], r[cols.index('nmse')], r[cols.index('nmse_abl')]))
    return rows, z


def summarize(rows):
    """rows: (task, ep, start, sigma, full, cut) for one sigma. Returns point estimates and episode-bootstrap CIs."""
    eps = sorted({(r[0], r[1]) for r in rows})
    by_ep = {e: [] for e in eps}
    for r in rows:
        by_ep[(r[0], r[1])].append((r[4], r[5]))
    sums = np.array([[sum(x[0] for x in by_ep[e]), sum(x[1] for x in by_ep[e]), len(by_ep[e])] for e in eps])
    full, cut = sums[:, 0].sum() / sums[:, 2].sum(), sums[:, 1].sum() / sums[:, 2].sum()
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(eps), size=(N_BOOT, len(eps)))
    s = sums[idx].sum(1)                                    # (N_BOOT, 3)
    bf, bc = s[:, 0] / s[:, 2], s[:, 1] / s[:, 2]
    ci = lambda x: (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))
    return dict(n_episodes=len(eps), n_windows=int(sums[:, 2].sum()), nmse_full=full, nmse_full_ci=ci(bf),
                nmse_cut=cut, nmse_cut_ci=ci(bc), gain=cut / full, gain_ci=ci(bc / bf))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', type=Path, default=Path('artifacts/action_read/split.json'))
    p.add_argument('--out', type=Path, default=Path('artifacts/action_read/results/step0_baseline.csv'))
    p.add_argument('--rerun', type=Path, default=None, help='second run of M1 on ID: checked for bit-identical abl rows')
    a = p.parse_args()
    split = json.loads(a.split.read_text())
    out_rows = []
    for set_name, (tag, variant) in SETS.items():
        eval_eps = {t: set(v['eval']) for t, v in split['sets'][set_name]['tasks'].items()}
        for mk, mname in MODELS.items():
            src = Path(f'artifacts/attention/{mname}_{tag}.npz')
            rows, z = load_rows(src, variant, eval_eps)
            for sg in sorted({r[3] for r in rows}):
                s = summarize([r for r in rows if r[3] == sg])
                out_rows.append(dict(model=mk, checkpoint=mname, set=set_name, variant=variant, action_sigma=round(float(sg), 4),
                                     video_sigma=round(float(z['video_sigma']), 4), n_episodes=s['n_episodes'], n_windows=s['n_windows'],
                                     nmse_full=s['nmse_full'], nmse_full_lo=s['nmse_full_ci'][0], nmse_full_hi=s['nmse_full_ci'][1],
                                     nmse_cut=s['nmse_cut'], nmse_cut_lo=s['nmse_cut_ci'][0], nmse_cut_hi=s['nmse_cut_ci'][1],
                                     gain=s['gain'], gain_lo=s['gain_ci'][0], gain_hi=s['gain_ci'][1],
                                     noise_seed=0, boot_seed=BOOT_SEED, n_boot=N_BOOT, split_seed=split['seed'], source=str(src)))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys())); w.writeheader(); w.writerows(out_rows)
    print(f'wrote {a.out} ({len(out_rows)} rows)')
    for r in out_rows:
        print(f"{r['model']} {r['set']:3s} sigma {r['action_sigma']:.2f}: full {r['nmse_full']:.5f} [{r['nmse_full_lo']:.5f},{r['nmse_full_hi']:.5f}] "
              f"cut {r['nmse_cut']:.4f} gain {r['gain']:.1f} [{r['gain_lo']:.1f},{r['gain_hi']:.1f}]  ({r['n_episodes']} ep, {r['n_windows']} win)")
    if a.rerun is not None:
        z1 = np.load(f'artifacts/attention/{MODELS["M1"]}_aloha-c2c.npz', allow_pickle=True); z2 = np.load(a.rerun, allow_pickle=True)
        same_meta = np.array_equal(z1['abl_meta'], z2['abl_meta'])
        same = same_meta and np.array_equal(z1['abl'], z2['abl'])
        diff = float(np.abs(z1['abl'] - z2['abl']).max()) if z1['abl'].shape == z2['abl'].shape else float('nan')
        print(f'rerun check (M1, ID, all 1500 windows): windows identical {same_meta}, abl rows bit-identical {same}, max |diff| {diff:.3g}')


if __name__ == '__main__':
    main()
