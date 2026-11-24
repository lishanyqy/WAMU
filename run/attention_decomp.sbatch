#!/bin/bash
#SBATCH --job-name=attn_dec
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --exclude=p2-gpu-23,p2-gpu-25
#SBATCH --output=/home/a1905721/workspace/OpenWAM/artifacts/logs/slurm/attn_dec_%x_%j.log
# Exact joint-softmax entropy decomposition for one self-attention checkpoint on the aligned rand windows (sigma_a 0.5).
# usage: sbatch run/attention_decomp.sbatch <checkpoint name in ~/modelspace>
set -u
M=$1
export HF_HUB_OFFLINE=1 USE_TF=0 PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v CUDAcompat | paste -sd: -)
PY=/home/a1905721/.conda/envs/fastwam/bin/python
$PY -c 'import torch,sys; ok=torch.cuda.is_available(); print("torch cuda", ok); sys.exit(0 if ok else 3)' || { echo "BAD NODE $(hostname)"; exit 3; }
echo "node $(hostname) start $(date) model $M"
cd /home/a1905721/workspace/OpenWAM
$PY -u scripts/attention_video_share.py --ckpt-dir /home/a1905721/modelspace/$M --embodiment aloha-agilex \
    --dataset-dir /home/a1905721/dataspace/openwam-attn-aligned --variant randomized_500 --episodes-per-task 10 --windows 3 \
    --action-t 0.5 --tag aloha-c2r-dec 2>&1 | grep --line-buffered -v "Warning\|warn\|INFO\|normalizer\|Scene info\|instruction files"
/home/a1905721/.conda/envs/d4rt/bin/python - "$M" <<'PY'
import sys, numpy as np
z = np.load(f'artifacts/attention/{sys.argv[1]}_aloha-c2r-dec.npz', allow_pickle=True)
s = z['dec_hb'] + z['dec_g_hv'] + z['dec_1mg_ha']; err = np.abs(s - z['entropy']).max()
print(f'CHECK {sys.argv[1]}: max |H_b + gH_v + (1-g)H_a - H_all| = {err:.2e}; gamma {z["dec_gamma"].mean():.3f}')
PY
echo "end $(date)"
