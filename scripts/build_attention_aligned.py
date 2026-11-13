#!/usr/bin/env python
"""Symlink farm for the attention experiment on exactly the probe test episodes of clean2rand:
randomized_500: the 500 replayed episodes (wamu/random_gt_episodes.json, 10 per task) -> agilex_c2r test;
clean_50: the 500 clean test episodes of agilex_c2c (artifacts/downstream/split_c2c.json).
Official hdf5 / instructions / scene_info are linked, so scripts/attention_video_share.py reads only these episodes."""
import json
from collections import defaultdict
from pathlib import Path

HOME = Path.home(); ROOT = Path(__file__).resolve().parents[1]
SRC = HOME / 'dataspace/openwam/benchmark_data/robotwin2.0/dataset'
OUT = HOME / 'dataspace/openwam-attn-aligned'
sets = {'randomized_500': {t: [f'episode{e}' for e in eps] for t, eps in
                           json.loads((ROOT / 'wamu/random_gt_episodes.json').read_text())['embodiments']['agilex']['tasks'].items()}}
clean = defaultdict(list)
for k in json.loads((ROOT / 'artifacts/downstream/split_c2c.json').read_text())['agilex_c2c']['test']:
    t, e = k.split('__'); clean[t].append(e)
sets['clean_50'] = dict(clean)
for var, tasks in sets.items():
    n = 0
    for t, eps in tasks.items():
        s, d = SRC / t / f'aloha-agilex_{var}', OUT / t / f'aloha-agilex_{var}'
        (d / 'data').mkdir(parents=True, exist_ok=True)
        for x in ('instructions', 'scene_info.json'):
            if not (d / x).exists(): (d / x).symlink_to(s / x)
        for e in eps:
            if not (d / 'data' / f'{e}.hdf5').exists(): (d / 'data' / f'{e}.hdf5').symlink_to(s / 'data' / f'{e}.hdf5')
            n += 1
    print(var, len(tasks), 'tasks', n, 'episodes')
