#!/bin/bash
#SBATCH --job-name=wamu_real_smoke
#SBATCH --account=acvt
#SBATCH --partition=a100,newa100,singlegpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --exclude=p2-gpu-23,p2-gpu-25
#SBATCH --output=/home/a1905721/workspace/OpenWAM/logs/%x_%j.log
# End-to-end OpenWAM training smoke on lishanyang/wamu_real: real data, real
# Wan2.2-TI2V-5B weights, a few optimizer steps, and a checkpoint save.
# Single GPU: by default the AdamW states are offloaded to CPU so the 5B model
# fits on a 40/80GB A100. On H200 (141GB) use OFFLOAD=none.
# Without Slurm:  OFFLOAD=none bash run/wamu_real_smoke.sbatch
source ~/.bashrc
module load CUDAcompat/13.1-590.48.01 2>/dev/null
conda activate openwam
cd /home/a1905721/workspace/OpenWAM
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
echo "$(date) host $(hostname) commit $(git rev-parse --short HEAD 2>/dev/null)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print('torch', torch.__version__, 'cuda ok')" || exit 1

torchrun --nproc_per_node=1 --master_port=$((29500 + RANDOM % 1000)) scripts/train.py \
  --config-name train_robodojo_real.yaml \
  model.video_backbone.model_path=assets/video_backbone_ckpt/Wan2.2-TI2V-5B \
  training.debug=true \
  training.max_steps=${MAX_STEPS:-4} \
  training.save_steps=${MAX_STEPS:-4} \
  training.offload_optimizer_device=${OFFLOAD:-cpu} \
  training.output_path=outputs/wamu_real_smoke \
  "$@"
rc=$?; echo "$(date) exit $rc"; exit $rc
