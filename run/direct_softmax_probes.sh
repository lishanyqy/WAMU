#!/bin/bash
# clean2rand probes (rand only) for cross_attn_direct and cross_attention_softmax: extract agilex_c2r
# (250 clean train + 500 rand test) for each, then dense / flow / cls probes; rows are appended to the c2r layers.csv files.
cd /home/a1905721/workspace/OpenWAM
MODELS="cross_attn_direct cross_attention_softmax"; L=artifacts/logs/direct_softmax_probes.log; F=/home/a1905721/dataspace/openwam_gt_features
mkdir -p artifacts/logs/openwam_c2r
for M in $MODELS; do
    for attempt in 1 2; do   # resumable: finished episodes are skipped
        bash -c "source ~/.bashrc; module load CUDAcompat/12.2-535.161.08; conda activate fastwam; cd /home/a1905721/workspace/OpenWAM; \
          python -u scripts/extract_openwam_features.py --ckpt-dir /home/a1905721/modelspace/$M --wan-dir assets/video_backbone_ckpt/Wan2.2-TI2V-5B \
            --manifest artifacts/agilex_c2r_manifest.jsonl --output $F/agilex_c2r/$M --device cuda:0 --seed 0 --lean" >> artifacts/logs/openwam_c2r/$M.log 2>&1
        n=$(ls $F/agilex_c2r/$M 2>/dev/null | grep -c npz); echo "$(date '+%F %T') extracted $M attempt $attempt files=$n" >> $L
        [ "$n" -ge 750 ] && break
    done
done
for w in dense flow cls; do
    bash run/probe_c2r.sh $w $MODELS >> artifacts/logs/direct_softmax_probe_$w.log 2>&1
    echo "$(date '+%F %T') probes c2r $w exit=$?" >> $L
done
echo "$(date '+%F %T') ALL DONE" >> $L
