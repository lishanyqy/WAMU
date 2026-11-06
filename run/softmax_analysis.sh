#!/bin/bash
# cross_attention_softmax: attention share (Franka, then Aloha randomized) + Franka video quality; then rebuild the tables.
source ~/.bashrc; module load CUDAcompat/12.2-535.161.08; cd /home/a1905721/workspace/OpenWAM
L=artifacts/logs/softmax_analysis.log; M=cross_attention_softmax; D4RT=/home/a1905721/.conda/envs/d4rt/bin/python
export USE_TF=0 PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
until grep -q "ALL DOWNLOADS DONE" artifacts/logs/download_wams_softmax.log 2>/dev/null; do
    screen -ls | grep -q "\.dl_wams\b" || { echo "$(date) download screen died" >> $L; exit 1; }; sleep 60; done
[ -f ~/modelspace/$M/checkpoint_step_30000.safetensors ] || { echo "$(date) checkpoint missing" >> $L; exit 1; }
echo "$(date) == attention, Franka" >> $L
MODELS=$M ACTION_T='0.25 0.5 0.75' NEED=14 bash run/attention_sweep.sh
tail -n 5 artifacts/logs/attention_sweep.log >> $L
$D4RT scripts/build_attention_report.py franka v2 >> $L 2>&1
echo "$(date) == video quality, Franka" >> $L
for attempt in 1 2 3; do
    /home/a1905721/.conda/envs/fastwam/bin/python -u scripts/video_gen_quality.py --ckpt-dir ~/modelspace/$M --episodes-per-task 2 2>&1 | grep --line-buffered -v "Warning\|warn\|INFO\|VAE decoding" >> $L
    [ -f artifacts/videogen/${M}_franka.npz ] && break; sleep 300
done
$D4RT scripts/build_videogen_report.py franka >> $L 2>&1
echo "$(date) == attention, Aloha randomized" >> $L
EMB=aloha-agilex TAG=aloha-random DATASET=/home/a1905721/dataspace/openwam/benchmark_data/robotwin2.0/dataset EXTRA='--variant randomized_500 --samples-per-task 100' \
    MODELS=$M ACTION_T='0.25 0.5 0.75' NEED=14 bash run/attention_sweep.sh
tail -n 5 artifacts/logs/attention_sweep.log >> $L
$D4RT scripts/build_attention_report.py aloha-random v2 >> $L 2>&1
echo "$(date) softmax analysis complete" >> $L
