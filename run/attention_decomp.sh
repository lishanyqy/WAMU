#!/bin/bash
# Exact joint-softmax entropy decomposition (H_all = H_b + g*H_video + (1-g)*H_action) and video/action log-odds
# for the self-attention family on the aligned rand probe test episodes (sigma_a = 0.5).
cd /home/a1905721/workspace/OpenWAM
L=artifacts/logs/attention_c2r.log
for M in sft_from_scratch_robotwin_clean sft_robot_only_robotwin_clean sft_ego_robot_cotrain_robotwin_clean_action_sees_video self_attn_detach_robotwin_clean; do
    echo "$(date) == decomposition $M" >> $L
    EMB=aloha-agilex TAG=aloha-c2r-dec DATASET=/home/a1905721/dataspace/openwam-attn-aligned EXTRA="--variant randomized_500 --episodes-per-task 10" MODELS=$M ACTION_T='0.5' NEED=14 bash run/attention_sweep.sh
    /home/a1905721/.conda/envs/d4rt/bin/python - "$M" >> $L 2>&1 <<'PY' || { echo "$(date) decomposition check failed" >> $L; exit 1; }
import sys, numpy as np
z = np.load(f'artifacts/attention/{sys.argv[1]}_aloha-c2r-dec.npz', allow_pickle=True)
s = z['dec_hb'] + z['dec_g_hv'] + z['dec_1mg_ha']; err = np.abs(s - z['entropy']).max()
print(f'{sys.argv[1]}: max |H_b + gH_v + (1-g)H_a - H_all| over layers = {err:.2e}; gamma {z["dec_gamma"].mean():.3f}')
assert err < 1e-3
PY
done
echo "$(date) decomposition done" >> $L
