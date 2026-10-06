#!/bin/bash
# Per-layer dense downstream probes (depth + target segmentation) on unpooled tokens,
# 10% train split, few-shot data, 60 epochs (near-converged dense heads). Resumable: finished (model, layer, task) rows are skipped.
# Usage: bash run/probe_dense_layers.sh <openwam|videomaev2|dinov2> <agilex|franka> [...]
source ~/.bashrc
module load CUDAcompat/12.2-535.161.08
cd /home/a1905721/workspace/OpenWAM
# Four of these jobs share a 72-core node with other users; uncapped BLAS/OMP pools oversubscribe it badly.
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12
what=$1; shift
for e in ${@:-agilex franka}; do
    case $what in
        openwam)    /home/a1905721/.conda/envs/d4rt/bin/python -u scripts/probe_dense_layers.py --embodiment $e --refs d4rt --epochs 60 --output artifacts/downstream/dense_layers_ep60 --group ${GROUP:-5} ;;
        videomaev2) conda deactivate; conda activate videomae; export LD_PRELOAD=/usr/lib64/libgnutls.so.30
                    python -u scripts/probe_dense_layers.py --embodiment $e --refs videomaev2 --models --group 3 --epochs 60 --output artifacts/downstream/dense_layers_ep60 --csv-name layers_videomaev2.csv ;;
        dinov2)     conda deactivate; conda activate fastwam
                    python -u scripts/probe_dense_layers.py --embodiment $e --refs dinov2 --models --group ${GROUP:-5} --epochs 60 --output artifacts/downstream/dense_layers_ep60 --csv-name layers_dinov2.csv ;;
    esac
done
