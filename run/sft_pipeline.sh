#!/bin/bash
# Add sft_ego_robot_cotrain_robotwin_full_action_sees_video to the study, after the current flow sweeps release the GPU:
#   1. extract its unpooled head tokens for both robots (same canvas/seed as the other 7; ~3 h + 637 GB per robot)
#   2. run every per-layer probe on it: task classification (10 ep), depth+segmentation (60 ep), flow/tracking (60 ep)
#   3. rebuild the HTML and PDF reports
# Every step is resumable; rerun this script to continue after an interruption.
cd /home/a1905721/workspace/OpenWAM
V=sft_ego_robot_cotrain_robotwin_full_action_sees_video
log() { echo "$(date '+%F %T') $*"; }
until ! screen -ls | grep -qE '\.flow_wam_(agilex|franka)'; do sleep 600; done
log "flow OpenWAM sweeps finished; starting extraction of $V"
bash run/extract_openwam_gt_parallel.sh 1 $V
for e in agilex franka; do n=$(ls /home/a1905721/dataspace/openwam_gt_features/$e/$V 2>/dev/null | grep -c npz); want=$([ $e = agilex ] && echo 2500 || echo 2300); [ "$n" -ge "$want" ] || { log "extraction incomplete for $e ($n/$want); stopping"; exit 1; }; done
log "extraction complete; running probes"
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
bash run/probe_task_cls_layers.sh agilex franka                 # resumable: only the new model is missing
bash run/probe_dense_layers.sh openwam agilex franka             # idem (writes to dense_layers_ep60)
bash run/flow_guarded.sh openwam 40 241 agilex                   # 1 D4RT + 8 x 30 rows per robot
bash run/flow_guarded.sh openwam 40 241 franka
log "probes done; rebuilding reports"
/home/a1905721/.conda/envs/d4rt/bin/python scripts/build_probe_report.py
/home/a1905721/.conda/envs/d4rt/bin/python scripts/build_probe_pdf.py 2>&1 | grep -v -i warn | tail -1
log "sft pipeline complete"
