#!/bin/bash
# cross_attention_softmax on the aligned rand probe test episodes: full attention run (3 action sigmas, aloha-c2r),
# then the per-frame / video-only entropy run (sigma_a 0.5, aloha-c2r-ent) as for the other seven models.
cd /home/a1905721/workspace/OpenWAM
L=artifacts/logs/attention_c2r.log; M=cross_attention_softmax; FARM=/home/a1905721/dataspace/openwam-attn-aligned
for spec in "aloha-c2r 0.25_0.5_0.75" "aloha-c2r-ent 0.5"; do set -- $spec
    echo "$(date) == $M ($1)" >> $L
    EMB=aloha-agilex TAG=$1 DATASET=$FARM EXTRA="--variant randomized_500 --episodes-per-task 10" MODELS=$M ACTION_T="${2//_/ }" NEED=14 bash run/attention_sweep.sh
done
/home/a1905721/.conda/envs/d4rt/bin/python scripts/build_attention_report.py aloha-c2r c2r >> $L 2>&1
echo "$(date) $M attention c2r done" >> $L
