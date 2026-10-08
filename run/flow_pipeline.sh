#!/bin/bash
# Waits for the _extra download to finish, builds the dense-flow GT for both embodiments, then launches the
# per-layer flow/tracking sweeps: OpenWAM (+D4RT) on both robots and VideoMAEv2 at once, DINOv2 after VideoMAEv2
# (at most three token-heavy jobs resident, which fit the node's RAM).
cd /home/a1905721/workspace/OpenWAM
until grep -q DONE /home/a1905721/dataspace/openwam-robotwin-gt.download3.log 2>/dev/null; do sleep 60; done
echo "$(date) download complete; building flow GT"
export OMP_NUM_THREADS=4
/home/a1905721/.conda/envs/d4rt/bin/python scripts/build_flow_gt.py 2>&1 | grep -v -i warn
ls -la artifacts/downstream/gt/*_flow.npz || { echo "GT build failed"; exit 1; }
mkdir -p artifacts/logs
screen -dmS flow_wam_agilex bash -c "bash run/probe_flow_layers.sh openwam agilex > artifacts/logs/probe_flow_openwam_agilex.log 2>&1"
screen -dmS flow_wam_franka bash -c "bash run/probe_flow_layers.sh openwam franka > artifacts/logs/probe_flow_openwam_franka.log 2>&1"
screen -dmS flow_vmae bash -c "bash run/probe_flow_layers.sh videomaev2 agilex franka > artifacts/logs/probe_flow_videomaev2.log 2>&1"
screen -dmS flow_dino bash -c "while screen -ls | grep -q '\.flow_vmae'; do sleep 300; done; bash run/probe_flow_layers.sh dinov2 agilex franka > artifacts/logs/probe_flow_dinov2.log 2>&1"
echo "$(date) sweeps launched"; screen -ls | grep flow_
