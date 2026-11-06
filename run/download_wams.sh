#!/bin/bash
# Fetch the four new checkpoints (step 30000 + config + stats + tokenizer) into ~/modelspace/<short name>/
export HF_HUB_DISABLE_XET=1
PY=/home/a1905721/.conda/envs/openwam/bin/python
$PY - <<'PYEOF'
import shutil, os
from pathlib import Path
from huggingface_hub import snapshot_download
out = Path.home() / 'modelspace'
jobs = [('sft_from_scratch_robotwin_clean', 'OpenWAM/sft_from_scratch_robotwin_clean', 'model', None),
        ('cross_attn_clean', 'paperupload/WAMs', 'dataset', 'action_film/cross_attn_clean/2026-10-14_17-14-10'),
        ('cross_attn_detach_robotwin_clean', 'paperupload/WAMs', 'dataset', 'action_film/cross_attn_detach_robotwin_clean/2026-10-17_13-48-31'),
        ('cross_attn_clean_fix3072', 'paperupload/WAMs', 'dataset', 'action_film/cross_attn_clean_fix3072/2026-10-24_11-45-18'),
        ('cross_attention_softmax', 'paperupload/WAMs', 'dataset', 'action_film/cross_attention_softmax/2026-11-01_07-44-53')]
for name, repo, rtype, sub in jobs:
    dst = out / name; dst.mkdir(parents=True, exist_ok=True)
    if (dst / 'checkpoint_step_30000.safetensors').exists() and (dst / 'config.yaml').exists():
        print(f'== {name}\n   files: already present', flush=True); continue
    pats = ['config.yaml', 'normalization_stats.npy', 'checkpoint_step_30000.safetensors', 'tokenizer/*', 'tokenizer/**']
    allow = [f'{sub}/{p}' for p in pats] if sub else pats
    print(f'== {name}', flush=True)
    local = snapshot_download(repo, repo_type=rtype, allow_patterns=allow, local_dir=str(dst / '_dl'))
    src = Path(local) / sub if sub else Path(local)
    for item in src.iterdir():
        target = dst / item.name
        if target.exists() or target.is_symlink():
            continue
        shutil.move(str(item), str(target))
    print(f'   files: {sorted(p.name for p in dst.iterdir())}', flush=True)
print('ALL DOWNLOADS DONE', flush=True)
PYEOF
