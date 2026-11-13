#!/bin/bash
# Attention share / entropy / video ablation for the five clean2rand models on exactly the probe test episodes
# (scripts/build_attention_aligned.py): 500 clean (agilex_c2c test) and 500 randomized (agilex_c2r test), 3 windows each (start/middle/end).
cd /home/a1905721/workspace/OpenWAM
L=artifacts/logs/attention_c2r.log
FARM=/home/a1905721/dataspace/openwam-attn-aligned
ALL="sft_from_scratch_robotwin_clean cross_attn_clean cross_attn_detach_robotwin_clean self_attn_detach_robotwin_clean sft_ego_robot_cotrain_robotwin_clean_action_sees_video"
D4RT=/home/a1905721/.conda/envs/d4rt/bin/python
for spec in "randomized_500 aloha-c2r" "clean_50 aloha-c2c"; do set -- $spec
    echo "$(date) == $1 ($2)" >> $L
    EMB=aloha-agilex TAG=$2 DATASET=$FARM EXTRA="--variant $1 --episodes-per-task 10" MODELS="$ALL" ACTION_T='0.25 0.5 0.75' NEED=14 bash run/attention_sweep.sh
    $D4RT scripts/build_attention_report.py $2 c2r >> $L 2>&1
done
echo "$(date) attention c2r (aligned) complete" >> $L
