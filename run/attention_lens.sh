#!/bin/bash
# Logit lens on the action stream (layers 20-29, with / without video) for three checkpoints, Franka.
cd /home/a1905721/workspace/OpenWAM
L=artifacts/logs/attention_lens.log
for m in robotwin_dual_system_joint_self_attention robotwin_dual_system_joint_cross_attention robotwin_dual_system_joint_cross_attention_detach; do
    [ -f artifacts/attention/${m}_franka_lens.npz ] && { echo "$(date) skip $m" >> $L; continue; }
    echo "$(date) == $m" >> $L
    bash run/attention_guarded.sh 14 --ckpt-dir ~/modelspace/$m --embodiment franka --episodes-per-task 2 --windows 3 --action-t 0.25 0.5 0.75 \
        --decode-layers 20 21 22 23 24 25 26 27 28 29 --output artifacts/attention/${m}_franka_lens.npz >> $L 2>&1
done
echo "$(date) lens sweep complete" >> $L
