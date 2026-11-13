#!/bin/bash
#SBATCH --account=acvt
#SBATCH --partition=icelake
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/home/a1905721/workspace/OpenWAM/artifacts/logs/action_read/%x_%j.log
# Generic CPU job for wamu/action_read_subspace_plan.md: runs scripts/action_read/<script> with the given args (fastwam env: torch + matplotlib + h5py).
cd /home/a1905721/workspace/OpenWAM
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONNOUSERSITE=1
echo "$(date) host $(hostname) commit $(git rev-parse --short HEAD) args: $*"
script=$1; shift
/home/a1905721/.conda/envs/fastwam/bin/python -u scripts/action_read/$script "$@"
rc=$?; echo "$(date) exit $rc"; exit $rc
