#!/bin/bash
# Fetch aloha-agilex_randomized_500.zip for all 50 RoboTwin 2.0 tasks (TianxingChen/RoboTwin2.0) and unzip each into
# benchmark_data/robotwin2.0/dataset/<task>/ (4 parallel downloads, resumable).
export HF_HUB_DISABLE_XET=1
ROOT=/home/a1905721/dataspace/openwam/benchmark_data/robotwin2.0/dataset
PY=/home/a1905721/.conda/envs/openwam/bin/python
$PY - <<'PYEOF'
import os, subprocess, zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
ROOT = Path('/home/a1905721/dataspace/openwam/benchmark_data/robotwin2.0/dataset')
api = HfApi(); tasks = sorted(t.path.split('/')[-1] for t in api.list_repo_tree('TianxingChen/RoboTwin2.0', path_in_repo='dataset', repo_type='dataset'))
def one(task):
    dst = ROOT / task / 'aloha-agilex_randomized_500'
    if (dst / 'data').is_dir() and len(list((dst / 'data').glob('episode*.hdf5'))) >= 500:
        return f'{task}: already present'
    z = hf_hub_download('TianxingChen/RoboTwin2.0', f'dataset/{task}/aloha-agilex_randomized_500.zip', repo_type='dataset', local_dir=str(ROOT / '_zips'))
    with zipfile.ZipFile(z) as zf:
        names = zf.namelist(); top = sorted({n.split('/')[0] for n in names})
        # the archive either carries the variant folder itself or its contents (data/, instructions/, scene_info.json)
        target = ROOT / task if 'aloha-agilex_randomized_500' in top else dst
        target.mkdir(parents=True, exist_ok=True); zf.extractall(target)
    os.remove(z)
    n = len(list((dst / 'data').glob('episode*.hdf5')))
    return f'{task}: unzipped, {n} episodes, top-level {top[:4]}'
with ThreadPoolExecutor(4) as ex:
    for msg in ex.map(one, tasks):
        print(msg, flush=True)
print('ALL RANDOMIZED DATA DONE', flush=True)
PYEOF
