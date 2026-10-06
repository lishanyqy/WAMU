#!/usr/bin/env python
"""Point an existing Robotwin manifest at the replayed GT HDF5 (lishanyang/openwam-robotwin-gt)."""
import argparse
import json
from pathlib import Path

import h5py


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True, help='old <embodiment>_robotwin10s_manifest.jsonl')
    p.add_argument('--gt-root', type=Path, required=True, help='openwam-robotwin-gt/<embodiment>')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--replay-failed', type=Path, default=None, help='reports/replay_success_check_failed.json')
    args = p.parse_args()
    failed = set()
    if args.replay_failed and args.replay_failed.exists():
        for x in json.loads(args.replay_failed.read_text()):
            if x.get('embodiment', args.gt_root.name) == args.gt_root.name:
                failed.add((x['task'], x['episode']))
    rows, missing = [], []
    for line in args.manifest.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        src = args.gt_root / row['task'] / 'gt_export' / 'data' / f"{row['episode']}.hdf5"
        if not src.exists():
            missing.append(str(src))
            continue
        with h5py.File(src, 'r') as f:
            total = len(f['observation/head_camera/rgb'])
        row.update(gt_hdf5=str(src.resolve()), gt_frames=total,
                   replay_success_failed=(row['task'], row['episode']) in failed)
        rows.append(row)
    if missing:
        raise SystemExit(f'{len(missing)} GT files missing, e.g. {missing[:3]}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    print(json.dumps({'rows': len(rows), 'replay_success_failed': sum(r['replay_success_failed'] for r in rows)}))


if __name__ == '__main__':
    main()
