#!/bin/bash
# Action->video attention share on Franka for every self-/cross-attention dual-system checkpoint (sequential, GPU-guarded).
cd /home/a1905721/workspace/OpenWAM
L=artifacts/logs/attention_sweep.log
for m in robotwin_dual_system_joint_self_attention robotwin_dual_system_joint_cross_attention robotwin_dual_system_joint_cross_attention_detach \
         robotwin_dual_system_joint_self_attention_mutual robotwin_dual_system_joint_self_attention_video_sees_action robotwin_dual_system_joint_self_attention_isolated \
         sft_ego_robot_cotrain_robotwin_full_action_sees_video; do
    [ -z "$FORCE" ] && [ -f artifacts/attention/${m}_${EMB:-franka}.npz ] && { echo "$(date) skip $m (done)" >> $L; continue; }
    echo "$(date) == $m" >> $L
    bash run/attention_guarded.sh ${NEED:-14} --ckpt-dir ~/modelspace/$m --embodiment ${EMB:-franka} --dataset-dir ${DATASET:-/home/a1905721/dataspace/robotwin2.0/dataset} --episodes-per-task 2 --windows 3 --action-t ${ACTION_T:-0.25 0.75} >> $L 2>&1
done
echo "$(date) attention sweep complete" >> $L
