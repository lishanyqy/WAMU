#!/bin/bash
# clean baseline for clean2rand: same clean train split, tested on 500 clean test episodes (embodiment agilex_c2c).
# Same settings as the clean report (dense/flow 60 epochs, task classification 10 epochs), all 30 layers.
# Usage: bash run/probe_c2r.sh <dense|flow|cls|all> [model ...]   (default: the five clean2rand models)
source ~/.bashrc
module load CUDAcompat/12.2-535.161.08
cd /home/a1905721/workspace/OpenWAM
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12
PY=/home/a1905721/.conda/envs/d4rt/bin/python
what=$1; shift
models=("$@")
[ ${#models[@]} -eq 0 ] && models=(sft_from_scratch_robotwin_clean cross_attn_clean cross_attn_detach_robotwin_clean
                                   self_attn_detach_robotwin_clean sft_ego_robot_cotrain_robotwin_clean_action_sees_video)
common=(--embodiment agilex_c2c --split artifacts/downstream/split_c2c.json --models "${models[@]}" --refs)
O=artifacts/downstream/c2c
[[ $what == dense || $what == all ]] && $PY -u scripts/probe_dense_layers.py "${common[@]}" --epochs 60 --output $O/dense_layers_ep60 --group ${GROUP:-5}
[[ $what == flow  || $what == all ]] && $PY -u scripts/probe_flow_layers.py  "${common[@]}" --epochs 60 --output $O/flow_layers_ep60 --group ${GROUP:-3}
[[ $what == cls   || $what == all ]] && $PY -u scripts/probe_task_cls_layers.py "${common[@]}" --epochs 10 --output $O/task_cls_layers_ep10 --group ${GROUP:-5}
