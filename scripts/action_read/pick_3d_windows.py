#!/usr/bin/env python
"""Windows for the step-3d figures: EVAL episodes, window start 0 (in both the causal and the step-3 window sets),
the 3 lowest and 3 highest NMSE_full of M1 at action sigma 0.5; the same windows are drawn for every model."""
import json

import numpy as np

for s in ('ID', 'OOD'):
    z = np.load(f'artifacts/action_read/raw/causal_M1_{s}_R{s}.npz', allow_pickle=True)
    v = [str(x) for x in z['variants']]
    full = z['nmse'][:, v.index('full')]
    sel = np.isclose(z['action_sigma'], 0.5, atol=1e-2) & np.array([int(k[2]) == 0 for k in z['key']])
    idx = np.nonzero(sel)[0]
    o = idx[np.argsort(full[idx])]
    pick = [(i, 'lowest NMSE_full (M1)') for i in o[:3]] + [(i, 'highest NMSE_full (M1)') for i in o[-3:]]
    out = [dict(task=str(z['key'][i][0]), episode=int(z['key'][i][1]), start=int(z['key'][i][2]), why=w) for i, w in pick]
    json.dump(out, open(f'artifacts/action_read/figs/step3d_windows_{s}.json', 'w'), indent=1)
print('3d windows written')
