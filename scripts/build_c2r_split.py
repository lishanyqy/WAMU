#!/usr/bin/env python
"""clean2rand probe setup (wamu/clean2rand_analysis.md): probes train on the clean Agilex 10% split and are tested on
the replayed randomized GT (wamu/random_gt_episodes.json). Exposes both as one virtual embodiment `agilex_c2r`:

  GT    openwam-robotwin-gt/agilex_c2r/<task>/gt_export/data/episode<N>(_extra).hdf5     -> clean GT (train + episode0)
                                                       randepisode<N>(_extra).hdf5 -> randomized GT (test)
  split artifacts/downstream/split_c2r.json  {"agilex_c2r": {"train": clean train keys, "test": rand keys}}
  rows  artifacts/agilex_c2r_manifest.jsonl  (extract_openwam_features.py input; wrist cameras from the official hdf5,
        as for the clean features)
Keys keep the task prefix, so the task-classification labels are shared between clean train and rand test.
"""
import json
from pathlib import Path

HOME = Path.home()
CLEAN = HOME / 'dataspace/openwam-robotwin-gt/agilex'
RAND = HOME / 'dataspace/openwam-robotwin-gt-random/agilex'
OUT = HOME / 'dataspace/openwam-robotwin-gt/agilex_c2r'
SRC = HOME / 'dataspace/openwam/benchmark_data/robotwin2.0/dataset'
EMB = 'agilex_c2r'


def link(dst, src):
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src)


def main():
    split = json.loads(Path('artifacts/downstream/split.json').read_text())['agilex']
    plan = json.loads(Path('wamu/random_gt_episodes.json').read_text())['embodiments']['agilex']['tasks']
    clean_rows = {}
    for line in Path('artifacts/agilex_robotwin_gt_manifest.jsonl').read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            clean_rows[f"{r['task']}__{r['episode']}"] = r
    train = list(split['train'])
    rows = []
    for key in train + [f'{t}__episode0' for t in plan]:
        t, ep = key.split('__')
        for suf in ('', '_extra'):
            link(OUT / t / 'gt_export/data' / f'{ep}{suf}.hdf5', CLEAN / t / 'gt_export/data' / f'{ep}{suf}.hdf5')
    for key in train:
        r = dict(clean_rows[key])
        r['gt_hdf5'] = str(OUT / r['task'] / 'gt_export/data' / f"{r['episode']}.hdf5")
        rows.append(r)
    test = []
    for t, eps in plan.items():
        for e in eps:
            key = f'{t}__randepisode{e}'
            test.append(key)
            for suf in ('', '_extra'):
                link(OUT / t / 'gt_export/data' / f'randepisode{e}{suf}.hdf5', RAND / t / 'gt_export_randomized/data' / f'episode{e}{suf}.hdf5')
            rows.append({'task': t, 'variant': 'aloha-agilex_randomized_500', 'episode': f'randepisode{e}',
                         'video': str(SRC / t / 'aloha-agilex_randomized_500/video' / f'episode{e}.mp4'),
                         'num_frames': 33, 'frame_stride': 4,
                         'gt_hdf5': str(OUT / t / 'gt_export/data' / f'randepisode{e}.hdf5')})
    Path('artifacts/downstream/split_c2r.json').write_text(json.dumps({'train_frac': None, 'seed': split.get('seed'), EMB: {'train': train, 'test': test}}, indent=1))
    Path(f'artifacts/{EMB}_manifest.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    print(f'{EMB}: train {len(train)} clean, test {len(test)} rand, manifest rows {len(rows)}')


if __name__ == '__main__':
    main()
