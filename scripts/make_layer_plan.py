#!/usr/bin/env python
"""Per-model layer plan for a re-run with another seed: for every embodiment, task and model, the TOP_K layers of the
seed-0 sweep by the task's primary metric (models with <= TOP_K layers keep all of them; DINOv2 is left out of flow).
Output: artifacts/downstream/layer_plan_top10.json  {embodiment: {task: {model: [layers]}}}
"""
import csv
import json
from pathlib import Path

TOP_K = 10
D = Path('artifacts/downstream')
# task -> (csv dir, primary metric, lower_is_better, task filter)
SRC = {'cls': (D / 'task_cls_layers_ep10', 'acc', False, None),
       'depth': (D / 'dense_layers_ep60', 'absrel', True, 'depth'),
       'seg': (D / 'dense_layers_ep60', 'iou', False, 'segmentation'),
       'flow': (D / 'flow_layers_ep60', 'epe3d', True, None)}


def main():
    plan = {}
    for emb in ('agilex', 'franka'):
        plan[emb] = {}
        for task, (d, met, lower, tfilt) in SRC.items():
            rows = []
            for f in sorted((d / emb).glob('layers*.csv')):
                if 'zeroshot' in f.name:
                    continue
                rows += [r for r in csv.DictReader(open(f)) if (tfilt is None or r['task'] == tfilt) and r[met] not in ('', None)]
            if task == 'flow':
                rows = [r for r in rows if r['model'] != 'dinov2']
            by = {}
            for r in rows:
                by.setdefault(r['model'], {})[int(r['layer'])] = float(r[met])
            plan[emb][task] = {}
            for m, v in by.items():
                if m == 'dinov2' and task == 'cls':
                    v = {l: x for l, x in v.items() if l != 0}          # patch-embedding CLS is constant (chance)
                top = sorted(sorted(v, key=lambda l: v[l], reverse=not lower)[:TOP_K])
                plan[emb][task][m] = top
            print(f'{emb} {task}: ' + ', '.join(f'{m.replace("robotwin_dual_system_", "")[:22]} {len(ls)}' for m, ls in plan[emb][task].items()))
    out = D / 'layer_plan_top10.json'
    out.write_text(json.dumps(plan, indent=1))
    print('wrote', out)


if __name__ == '__main__':
    main()
