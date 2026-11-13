#!/bin/bash
# Waits for run/extract_c2r.sh, smoke-tests the three probes on one layer of the first model, then runs all probes.
cd /home/a1905721/workspace/OpenWAM
F=/home/a1905721/dataspace/openwam_gt_features/agilex_c2r
first=sft_from_scratch_robotwin_clean
until [ "$(ls $F/$first 2>/dev/null | grep -c npz)" -ge 750 ]; do sleep 60; done
echo "$(date '+%F %T') smoke test on $first layer 15"
for w in dense flow cls; do
  bash -c "source ~/.bashrc; module load CUDAcompat/12.2-535.161.08; cd /home/a1905721/workspace/OpenWAM; \
    /home/a1905721/.conda/envs/d4rt/bin/python -u scripts/probe_$( [ $w = dense ] && echo dense || ([ $w = flow ] && echo flow || echo task_cls) )_layers.py \
    --embodiment agilex_c2r --split artifacts/downstream/split_c2r.json --models $first --refs --layers 15 --epochs 2 \
    --output artifacts/downstream/c2r/smoke_$w" > artifacts/logs/c2r_smoke_$w.log 2>&1
  echo "$(date '+%F %T') smoke $w exit=$?"; tail -3 artifacts/logs/c2r_smoke_$w.log
done
while pgrep -f "run/extract_c2r.sh" > /dev/null; do sleep 120; done
echo "$(date '+%F %T') extraction done: $(for d in $F/*/; do echo -n "$(basename $d)=$(ls $d | grep -c npz) "; done)"
for w in dense flow cls; do
  bash run/probe_c2r.sh $w > artifacts/logs/c2r_probe_$w.log 2>&1
  echo "$(date '+%F %T') probes $w exit=$?"
done
echo "$(date '+%F %T') ALL DONE"
