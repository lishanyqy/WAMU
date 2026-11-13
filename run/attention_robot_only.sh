#!/bin/bash
# Attention analysis (aligned c2r / c2c probe test episodes) for sft_robot_only_robotwin_clean, once its download is complete.
cd /home/a1905721/workspace/OpenWAM
L=artifacts/logs/attention_c2r.log; M=sft_robot_only_robotwin_clean; D=~/modelspace/$M
D4RT=/home/a1905721/.conda/envs/d4rt/bin/python
echo "$(date) == $M: waiting for download" >> $L
until [ -f $D/checkpoint_step_30000.safetensors ] && [ -f $D/config.yaml ] && ! pgrep -f "snapshot_download.*$M" > /dev/null; do sleep 60; done
python3 - "$D/checkpoint_step_30000.safetensors" >> $L 2>&1 <<'PY' || { echo "$(date) checkpoint unreadable" >> $L; exit 1; }
import json, struct, sys
f = open(sys.argv[1], 'rb'); n = struct.unpack('<Q', f.read(8))[0]; h = json.loads(f.read(n))
print('checkpoint header ok,', len(h), 'tensors')
PY
for spec in "randomized_500 aloha-c2r" "clean_50 aloha-c2c"; do set -- $spec
    echo "$(date) == $M $1 ($2)" >> $L
    EMB=aloha-agilex TAG=$2 DATASET=/home/a1905721/dataspace/openwam-attn-aligned EXTRA="--variant $1 --episodes-per-task 10" MODELS=$M ACTION_T='0.25 0.5 0.75' NEED=14 bash run/attention_sweep.sh
    $D4RT scripts/build_attention_report.py $2 c2r >> $L 2>&1
done
echo "$(date) $M attention done" >> $L
