#!/usr/bin/env python
"""Clean baseline for clean2rand: the same 250 clean train episodes as agilex_c2r, tested on 10 clean test episodes per
task (numpy seed 0 from the clean 90% test split; 500 episodes, as many as the randomized test). Virtual embodiment
`agilex_c2c`: GT symlinks under openwam-robotwin-gt/agilex_c2c, split artifacts/downstream/split_c2c.json, manifest
artifacts/agilex_c2c_manifest.jsonl (test episodes only; train features are symlinked from agilex_c2r)."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HOME = Path.home()
CLEAN = HOME / 'dataspace/openwam-robotwin-gt/agilex'
OUT = HOME / 'dataspace/openwam-robotwin-gt/agilex_c2c'
FEAT = HOME / 'dataspace/openwam_gt_features'
EMB = 'agilex_c2c'


def link(dst, src):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src)


def main():
    split = json.loads(Path('artifacts/downstream/split.json').read_text())['agilex']
    by_task = defaultdict(list)
    for k in split['test']:
        by_task[k.split('__')[0]].append(k)
    rng = np.random.default_rng(0)
    test = []
    for t in sorted(by_task):
        ks = sorted(by_task[t], key=lambda k: int(k.split('episode')[-1]))
        test += [ks[i] for i in sorted(rng.choice(len(ks), 10, replace=False))]
    train = list(split['train'])
    rows = {}
    for line in Path('artifacts/agilex_robotwin_gt_manifest.jsonl').read_text().splitlines():
        if line.strip():
            r = json.loads(line); rows[f"{r['task']}__{r['episode']}"] = r
    tasks = sorted({k.split('__')[0] for k in train + test})
    for key in train + test + [f'{t}__episode0' for t in tasks]:
        t, ep = key.split('__')
        for suf in ('', '_extra'):
            link(OUT / t / 'gt_export/data' / f'{ep}{suf}.hdf5', CLEAN / t / 'gt_export/data' / f'{ep}{suf}.hdf5')
    man = []
    for key in test:
        r = dict(rows[key]); r['gt_hdf5'] = str(OUT / r['task'] / 'gt_export/data' / f"{r['episode']}.hdf5"); man.append(r)
    Path('artifacts/downstream/split_c2c.json').write_text(json.dumps({'train_frac': None, 'seed': 0, EMB: {'train': train, 'test': test}}, indent=1))
    Path(f'artifacts/{EMB}_manifest.jsonl').write_text('\n'.join(json.dumps(r) for r in man) + '\n')
    print(f'{EMB}: train {len(train)}, test {len(test)} ({len(tasks)} tasks)')


def link_train_features(models):
    """agilex_c2c/<model>/<train key>.npz -> agilex_c2r/<model>/<train key>.npz (identical clean train episodes)."""
    train = json.loads(Path('artifacts/downstream/split_c2c.json').read_text())[EMB]['train']
    for m in models:
        for k in train:
            link(FEAT / EMB / m / f'{k}.npz', FEAT / 'agilex_c2r' / m / f'{k}.npz')


if __name__ == '__main__':
    import sys
    if sys.argv[1:2] == ['--link-train']:
        link_train_features(sys.argv[2:])
    else:
        main()
