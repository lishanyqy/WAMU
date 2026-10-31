#!/bin/bash
# Video generation quality on Franka for the four 30k-step checkpoints (after the aloha-random attention sweep).
source ~/.bashrc; module load CUDAcompat/12.2-535.161.08; cd /home/a1905721/workspace/OpenWAM
export USE_TF=0 PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
L=artifacts/logs/videogen.log; PY=/home/a1905721/.conda/envs/fastwam/bin/python
while screen -ls | grep -qE "\.(attn_v2_random|attn_v2_fixup)\b"; do sleep 120; done
echo "$(date) smoke test" >> $L
$PY -u scripts/video_gen_quality.py --ckpt-dir ~/modelspace/sft_from_scratch_robotwin_clean --limit-tasks 1 --episodes-per-task 1 --output /tmp/claude-1905721/videogen_smoke.npz 2>&1 | grep --line-buffered -v "Warning\|warn\|INFO" >> $L
grep -q "^done sft_from_scratch_robotwin_clean: 1 clips" $L || { echo "$(date) SMOKE FAILED, stopping" >> $L; exit 1; }
for m in sft_from_scratch_robotwin_clean cross_attn_clean cross_attn_detach_robotwin_clean cross_attn_clean_fix3072; do
    [ -f artifacts/videogen/${m}_franka.npz ] && continue
    echo "$(date) == $m" >> $L
    for attempt in 1 2 3; do
        $PY -u scripts/video_gen_quality.py --ckpt-dir ~/modelspace/$m --episodes-per-task 2 2>&1 | grep --line-buffered -v "Warning\|warn\|INFO" >> $L
        [ -f artifacts/videogen/${m}_franka.npz ] && break
        echo "$(date) attempt $attempt failed for $m" >> $L; sleep 300
    done
done
/home/a1905721/.conda/envs/d4rt/bin/python scripts/build_videogen_report.py franka >> $L 2>&1
echo "$(date) videogen sweep complete" >> $L
