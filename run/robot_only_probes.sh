#!/bin/bash
# clean2rand probes (rand only) for sft_robot_only_robotwin_clean: extract agilex_c2r (250 clean train + 500 rand test),
# then dense / flow / cls probes; rows are appended to the existing c2r layers.csv files. Resumable.
cd /home/a1905721/workspace/OpenWAM
M=sft_robot_only_robotwin_clean; L=artifacts/logs/robot_only_probes.log; F=/home/a1905721/dataspace/openwam_gt_features
while pgrep -f "extract_openwam_features.py --ckpt-dir /home/a1905721/modelspace/$M" > /dev/null; do sleep 60; done
if [ $(ls $F/agilex_c2r/$M | grep -c npz) -lt 750 ]; then   # finish any missing episodes (finished ones are skipped)
    bash -c "source ~/.bashrc; module load CUDAcompat/12.2-535.161.08; conda activate fastwam; cd /home/a1905721/workspace/OpenWAM; \
      python -u scripts/extract_openwam_features.py --ckpt-dir /home/a1905721/modelspace/$M --wan-dir assets/video_backbone_ckpt/Wan2.2-TI2V-5B \
        --manifest artifacts/agilex_c2r_manifest.jsonl --output $F/agilex_c2r/$M --device cuda:0 --seed 0 --lean" >> artifacts/logs/openwam_c2r/$M.log 2>&1
fi
echo "$(date '+%F %T') extracted c2r files=$(ls $F/agilex_c2r/$M | grep -c npz)" >> $L
for w in dense flow cls; do
    bash run/probe_c2r.sh $w $M >> artifacts/logs/robot_only_probe_c2r_$w.log 2>&1
    echo "$(date '+%F %T') probes c2r $w exit=$?" >> $L
done
echo "$(date '+%F %T') ALL DONE" >> $L
