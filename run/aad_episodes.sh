#!/bin/bash
# Per-episode AAD + action->video attention maps (scripts/aad_episode_dump.py) for the six Table-5 WAMs, then the figures.
source ~/.bashrc; module load CUDAcompat/12.2-535.161.08; cd /home/a1905721/workspace/OpenWAM
export USE_TF=0 PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
PY=/home/a1905721/.conda/envs/fastwam/bin/python; L=artifacts/logs/aad_episodes.log; OUT=${OUT:-artifacts/aad_episodes}
TASKS=${TASKS:-"beat_block_hammer handover_block stack_bowls_two open_laptop"}
MODELS=${MODELS:-"sft_from_scratch_robotwin_clean sft_robot_only_robotwin_clean sft_ego_robot_cotrain_robotwin_clean_action_sees_video cross_attn_detach_robotwin_clean cross_attn_clean cross_attn_direct"}
F() { grep --line-buffered -v "Warning\|warn\|INFO\|normalizer\|Scene info\|instruction files\|RoboTwinDataset"; }
for m in $MODELS; do
    [ -z "$FORCE" ] && [ -f $OUT/$m.npz ] && { echo "$(date) skip $m (done)" >> $L; continue; }
    echo "$(date) == $m" >> $L
    for attempt in 1 2; do
        $PY -u scripts/aad_episode_dump.py --ckpt $m --tasks $TASKS --episodes-per-task ${EPS:-3} --windows ${WINS:-3} --out-dir $OUT ${EXTRA:-} 2>&1 | F >> $L
        [ -f $OUT/$m.npz ] && break
    done
done
echo "$(date) aad episodes dump complete" >> $L
