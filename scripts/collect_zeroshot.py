#!/usr/bin/env python
"""Zero-shot specialist ceilings for the dense tasks, aggregated over the probe's TEST episodes (the 90% split) so the
numbers sit next to the probe rows on the same frames:
  sam_zeroshot  segmentation IoU of SAM ViT-H prompted with the GT target's box + centroid (oracle prompts)
  da3_zeroshot  depth AbsRel / delta1 of Depth Anything 3 after a per-frame least-squares scale + shift alignment to the GT
                (da3_zeroshot_scale: median scale only), and scale-free pairwise ordinal accuracy; needs <emb>_features_gt/da3_depth
Writes artifacts/downstream/dense_layers_ep60/<emb>/layers_zeroshot.csv in the dense-probe CSV layout (kind=zeroshot, layer 0).
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_task_cls_layers import split_keys

OUT = Path('artifacts/downstream/dense_layers_ep60')


def main():
    for emb in ('agilex', 'franka'):
        _, test = split_keys('artifacts/downstream/split.json', emb)
        rows = []
        sam = [np.load(f'artifacts/{emb}_features_gt/sam/{k}.npz')['zs_iou'] for k in test if Path(f'artifacts/{emb}_features_gt/sam/{k}.npz').exists()]
        if len(sam) == len(test):
            iou = np.concatenate(sam)
            rows.append(dict(model='sam_zeroshot', kind='zeroshot', layer=0, task='segmentation', tokens=0, absrel='', delta1='', pairwise='', iou=float(np.nanmean(iou)), epochs=0))
            print(f'{emb} sam_zeroshot: IoU {np.nanmean(iou):.4f} over {np.isfinite(iou).sum()} frames')
        else:
            print(f'{emb} sam: {len(sam)}/{len(test)} test episodes extracted, skipped')
        dd = Path(f'artifacts/{emb}_features_gt/da3_depth')
        gt = np.load(f'artifacts/downstream/gt/{emb}.npz')
        pos = {str(k): i for i, k in enumerate(gt['keys'])}
        gt_depth = gt['depth']                       # load once: indexing the NpzFile re-reads the whole array every time
        if all((dd / f'{k}.npz').exists() for k in test):
            # DA3 predicts scale-free depth: align each frame to the GT (a) by one median scale, (b) by least-squares scale + shift
            # (the MiDaS / Depth Anything protocol for relative depth, here in depth space); pairwise ordinal accuracy needs no alignment
            acc = {k: [] for k in ('absrel_s', 'delta1_s', 'absrel_a', 'delta1_a', 'pair')}
            for k in test:
                z = np.load(dd / f'{k}.npz'); pred = z['pred'].astype(np.float32); g = gt_depth[pos[k]][1:9].astype(np.float32)
                acc['pair'].append(z['zs_pairwise'])
                for j in range(8):
                    p_, g_ = pred[j].ravel(), g[j].ravel(); v = np.isfinite(g_) & (g_ > 0) & np.isfinite(p_) & (p_ > 0)
                    if v.sum() < 2:
                        continue
                    ps = p_[v] * np.median(g_[v] / p_[v])
                    A = np.stack([p_[v], np.ones(v.sum(), np.float32)], 1); (sc, sh), *_ = np.linalg.lstsq(A, g_[v], rcond=None); pa = np.maximum(p_[v] * sc + sh, 1e-3)
                    for tag, pp in (('s', ps), ('a', pa)):
                        acc[f'absrel_{tag}'].append(np.mean(np.abs(pp - g_[v]) / g_[v])); acc[f'delta1_{tag}'].append(np.mean(np.maximum(pp / g_[v], g_[v] / pp) < 1.25))
            pair = float(np.nanmean(np.concatenate(acc['pair'])))
            for model, tag in (('da3_zeroshot', 'a'), ('da3_zeroshot_scale', 's')):
                rows.append(dict(model=model, kind='zeroshot', layer=0, task='depth', tokens=0, absrel=float(np.mean(acc[f'absrel_{tag}'])),
                                 delta1=float(np.mean(acc[f'delta1_{tag}'])), pairwise=pair, iou='', epochs=0))
            print(f'{emb} da3 zero-shot: scale-only absrel {np.mean(acc["absrel_s"]):.4f} delta1 {np.mean(acc["delta1_s"]):.4f} | scale+shift absrel {np.mean(acc["absrel_a"]):.4f} delta1 {np.mean(acc["delta1_a"]):.4f} | pairwise {pair:.4f}')
        else:
            print(f'{emb} da3_depth: not all test episodes extracted, skipped')
        if rows:
            (OUT / emb).mkdir(parents=True, exist_ok=True)
            with open(OUT / emb / 'layers_zeroshot.csv', 'w', newline='') as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == '__main__':
    main()
