#!/bin/bash
# Per-frame and video-only attention entropy for the seven clean2rand models on the aligned rand episodes (sigma_a = 0.5 only).
cd /home/a1905721/workspace/OpenWAM
L=artifacts/logs/attention_c2r.log
ALL="sft_from_scratch_robotwin_clean cross_attn_direct cross_attn_clean cross_attn_detach_robotwin_clean self_attn_detach_robotwin_clean sft_ego_robot_cotrain_robotwin_clean_action_sees_video sft_robot_only_robotwin_clean"
echo "$(date) == entropy per frame (aloha-c2r-ent)" >> $L
EMB=aloha-agilex TAG=aloha-c2r-ent DATASET=/home/a1905721/dataspace/openwam-attn-aligned EXTRA="--variant randomized_500 --episodes-per-task 10" MODELS="$ALL" ACTION_T='0.5' NEED=14 bash run/attention_sweep.sh
echo "$(date) entropy per frame done" >> $L
