#!/bin/bash
# Per-layer dense 3D scene-flow / point-tracking probes (D4RT + TAPVid metrics) on unpooled tokens,
# 10% train split, 60 epochs. Resumable: finished (model, layer) rows are skipped.
# Usage: bash run/probe_flow_layers.sh <openwam|videomaev2|dinov2> <agilex|franka> [...]
source ~/.bashrc
module load CUDAcompat/12.2-535.161.08
cd /home/a1905721/workspace/OpenWAM
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12
what=$1; shift
for e in ${@:-agilex franka}; do
    case $what in
        openwam)    /home/a1905721/.conda/envs/d4rt/bin/python -u scripts/probe_flow_layers.py --embodiment $e --refs d4rt --group ${GROUP:-3} ;;
        videomaev2) conda deactivate; conda activate videomae; export LD_PRELOAD=/usr/lib64/libgnutls.so.30
                    python -u scripts/probe_flow_layers.py --embodiment $e --refs videomaev2 --models --group 1 --csv-name layers_videomaev2.csv ;;
        dinov2)     conda deactivate; conda activate fastwam
                    python -u scripts/probe_flow_layers.py --embodiment $e --refs dinov2 --models --group 2 --csv-name layers_dinov2.csv ;;
    esac
done
