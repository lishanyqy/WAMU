#!/bin/bash
#SBATCH --account=acvt
#SBATCH --partition=a100,newa100,singlegpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --exclude=p2-gpu-23,p2-gpu-25
#SBATCH --output=/home/a1905721/workspace/OpenWAM/artifacts/logs/action_read/%x_%j.log
# Generic single-GPU job for wamu/action_read_subspace_plan.md: runs scripts/action_read/<script> with the given args (fastwam env).
source ~/.bashrc
# CUDAcompat is needed on the login nodes (old 11.6 driver) and must NOT be used on compute nodes (CUDA error 803)
case "$(hostname)" in p2-log-*) module load CUDAcompat/12.2-535.161.08 2>/dev/null;;
  *) export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v CUDAcompat | paste -sd: -);; esac
cd /home/a1905721/workspace/OpenWAM
export USE_TF=0 PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
echo "$(date) host $(hostname) gpu $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) commit $(git rev-parse --short HEAD) args: $*"
script=$1; shift
/home/a1905721/.conda/envs/fastwam/bin/python -u scripts/action_read/$script "$@" 2>&1 | grep --line-buffered -v "Warning\|warn\|INFO"
rc=${PIPESTATUS[0]}; echo "$(date) exit $rc"; exit $rc
