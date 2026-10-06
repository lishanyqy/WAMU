#!/usr/bin/env python
"""Fixed probe split shared by layer selection and every downstream probe.

Per embodiment and per task, a seeded ``--train-frac`` of episodes goes to train and the
rest to test, so every task is represented in both and layer selection never sees test data.
Keys are ``<task>__<episode>``, matching the feature shard names.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifests', type=Path, nargs='+', required=True, help='<embodiment>_robotwin_gt_manifest.jsonl')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--train-frac', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    split = {'train_frac': args.train_frac, 'seed': args.seed}
    for manifest in args.manifests:
        embodiment = manifest.name.split('_')[0]
        by_task = defaultdict(list)
        replay_failed = []
        for line in manifest.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = f"{row['task']}__{row['episode']}"
            by_task[row['task']].append(key)
            if row.get('replay_success_failed'):
                replay_failed.append(key)
        rng = np.random.default_rng(args.seed)
        train, test = [], []
        for task in sorted(by_task):
            keys = sorted(by_task[task])
            n_train = max(1, int(round(len(keys) * args.train_frac)))
            pick = set(rng.choice(len(keys), n_train, replace=False).tolist())
            train += [k for i, k in enumerate(keys) if i in pick]
            test += [k for i, k in enumerate(keys) if i not in pick]
        split[embodiment] = {'train': train, 'test': test, 'replay_success_failed': sorted(replay_failed)}
        print(f'{embodiment}: tasks={len(by_task)} train={len(train)} test={len(test)} replay_failed={len(replay_failed)}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(split, indent=1))


if __name__ == '__main__':
    main()
