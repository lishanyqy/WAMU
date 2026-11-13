#!/bin/bash
# Clean baseline: waits for the clean2rand probes to finish, extracts the 500 clean test episodes for the five models
# (train features are symlinked from agilex_c2r), then runs the same probes on agilex_c2c.
cd /home/a1905721/workspace/OpenWAM
L=artifacts/logs/c2c_pipeline.log
while pgrep -f "run/probe_c2r.sh" > /dev/null || pgrep -f "c2r_resume.sh" > /dev/null; do sleep 300; done
echo "$(date '+%F %T') clean2rand probes finished; extracting clean test features" >> $L
models=(sft_from_scratch_robotwin_clean cross_attn_clean cross_attn_detach_robotwin_clean self_attn_detach_robotwin_clean sft_ego_robot_cotrain_robotwin_clean_action_sees_video)
mkdir -p artifacts/logs/openwam_c2c
for m in "${models[@]}"; do
    bash -c "source ~/.bashrc; module load CUDAcompat/12.2-535.161.08; conda activate fastwam; cd /home/a1905721/workspace/OpenWAM; \
      python -u scripts/extract_openwam_features.py --ckpt-dir /home/a1905721/modelspace/$m --wan-dir assets/video_backbone_ckpt/Wan2.2-TI2V-5B \
        --manifest artifacts/agilex_c2c_manifest.jsonl --output /home/a1905721/dataspace/openwam_gt_features/agilex_c2c/$m --device cuda:0 --seed 0 --lean" \
      > artifacts/logs/openwam_c2c/$m.log 2>&1
    echo "$(date '+%F %T') extracted $m exit=$? files=$(ls /home/a1905721/dataspace/openwam_gt_features/agilex_c2c/$m | grep -c npz)" >> $L
done
for w in dense flow cls; do
    bash run/probe_c2c.sh $w >> artifacts/logs/c2c_probe_$w.log 2>&1
    echo "$(date '+%F %T') probes $w exit=$?" >> $L
done
echo "$(date '+%F %T') ALL DONE" >> $L
