#!/bin/bash
# Cumulative video cut-off (video visible to the action stream only in layers < k, k = 20..29) for three checkpoints, Franka.
cd /home/a1905721/workspace/OpenWAM
L=artifacts/logs/attention_cut.log
for m in robotwin_dual_system_joint_self_attention robotwin_dual_system_joint_cross_attention robotwin_dual_system_joint_cross_attention_detach; do
    [ -f artifacts/attention/${m}_franka_cut.npz ] && { echo "$(date) skip $m" >> $L; continue; }
    echo "$(date) == $m" >> $L
    bash run/attention_guarded.sh 14 --ckpt-dir ~/modelspace/$m --embodiment franka --episodes-per-task 2 --windows 3 --action-t 0.25 0.5 0.75 \
        --cut-from 20 21 22 23 24 25 26 27 28 29 --output artifacts/attention/${m}_franka_cut.npz >> $L 2>&1
done
echo "$(date) cut sweep complete" >> $L
