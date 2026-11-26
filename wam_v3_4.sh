#PBS -l ngpus=4

#PBS -l ncpus=48

#PBS -l mem=200GB

#PBS -l jobfs=200GB

#PBS -q gpuhopper

#PBS -P at16

#PBS -l walltime=21:00:00

#PBS -l storage=scratch/ou29+gdata/zv32+scratch/zv32

#PBS -l wd
module load python3/3.12.13
module load cuda/12.9.0
source "/g/data/zv32/rz6525/conda3/etc/profile.d/conda.sh"
export CONDA_ENVS_PATH="/scratch/ou29/rz6525/conda3/envs/"
# Activate your environment
conda activate openwam
# export your robotwin data env
cd /g/data/zv32/rz6525/WAMU/

# export your WAMU real data env (default: datasets/lishanyang_wamu_real)
export OPEN_WAM_WAMU_REAL_DATA=/scratch/ou29/rz6525/WAMU/WAMU/assets/wamu_real


# finetune_ckpt_path: the run folder (highest step = 20000 is picked) or .../checkpoint_step_20000.safetensors
torchrun --nproc_per_node=4 scripts/train.py \
  --config-name train_wamu_real_pretrain_cotrain_multi_task.yaml \
  model.video_backbone.model_path=assets/video_backbone_ckpt/Wan2.2-TI2V-5B \
  training.finetune_ckpt_path=/scratch/zv32/rz6525/WAMU/outputs/wamu_real_pretrain_foundation/2026-11-21_14-15-41 \
  training.output_path=/scratch/zv32/rz6525/WAMU/outputs/wamu_real_pretrain_cotrain_multi_task