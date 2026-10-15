#!/bin/bash
# Re-run of every probe with seed 777 (new 10% split + probe seed) on each model's top-10 seed-0 layers
# (artifacts/downstream/layer_plan_top10.json). Outputs: artifacts/downstream/{task_cls_layers_ep10,dense_layers_ep60,flow_layers_ep60}_seed777.
# Sequential on the shared GPU: classification -> depth/segmentation -> flow/tracking. Every stage is resumable.
source ~/.bashrc
module load CUDAcompat/12.2-535.161.08
cd /home/a1905721/workspace/OpenWAM
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12
EXTRA="--seed 777 --split artifacts/downstream/split_seed777.json --layer-plan artifacts/downstream/layer_plan_top10.json"
D4RT=/home/a1905721/.conda/envs/d4rt/bin/python
L=artifacts/logs/seed777_pipeline.log
stage() { echo "$(date) == $*" >> $L; }

# ---- phase A: OpenWAM models only (user priority), classification -> depth/seg -> flow
stage "A1 classification, OpenWAM models"
for e in agilex franka; do
    $D4RT -u scripts/probe_task_cls_layers.py --embodiment $e --refs --epochs 10 --output artifacts/downstream/task_cls_layers_ep10_seed777 $EXTRA >> $L 2>&1
done
stage "A2 depth + segmentation, OpenWAM models"
for e in agilex franka; do
    $D4RT -u scripts/probe_dense_layers.py --embodiment $e --epochs 60 --group 5 --output artifacts/downstream/dense_layers_ep60_seed777 $EXTRA >> $L 2>&1
done
stage "A3 flow + tracking, OpenWAM models"
for e in agilex franka; do
    $D4RT -u scripts/probe_flow_layers.py --embodiment $e --group 1 --epochs 60 --output artifacts/downstream/flow_layers_ep60_seed777 $EXTRA >> $L 2>&1
done

# ---- phase B: reference encoders
stage "B1 classification, references"
for e in agilex franka; do
    $D4RT -u scripts/probe_task_cls_layers.py --embodiment $e --refs d4rt dinov2 --models --epochs 10 --output artifacts/downstream/task_cls_layers_ep10_seed777 $EXTRA >> $L 2>&1
done
conda deactivate; conda activate videomae; export LD_PRELOAD=/usr/lib64/libgnutls.so.30
for e in agilex franka; do
    python -u scripts/probe_task_cls_layers.py --embodiment $e --refs videomaev2 --models --group 3 --epochs 10 --output artifacts/downstream/task_cls_layers_ep10_seed777 --csv-name layers_videomaev2.csv $EXTRA >> $L 2>&1
done
unset LD_PRELOAD; conda deactivate

stage "B2 depth + segmentation, references"
for e in agilex franka; do
    $D4RT -u scripts/probe_dense_layers.py --embodiment $e --refs d4rt sam da3 --models --epochs 60 --group 5 --output artifacts/downstream/dense_layers_ep60_seed777 $EXTRA >> $L 2>&1
done
conda activate videomae; export LD_PRELOAD=/usr/lib64/libgnutls.so.30
for e in agilex franka; do
    python -u scripts/probe_dense_layers.py --embodiment $e --refs videomaev2 --models --group 3 --epochs 60 --output artifacts/downstream/dense_layers_ep60_seed777 --csv-name layers_videomaev2.csv $EXTRA >> $L 2>&1
done
unset LD_PRELOAD; conda deactivate; conda activate fastwam; export USE_TF=0 PYTHONNOUSERSITE=1
for e in agilex franka; do
    python -u scripts/probe_dense_layers.py --embodiment $e --refs dinov2 --models --group 5 --epochs 60 --output artifacts/downstream/dense_layers_ep60_seed777 --csv-name layers_dinov2.csv $EXTRA >> $L 2>&1
done
unset USE_TF PYTHONNOUSERSITE; conda deactivate

stage "B3 flow + tracking, references"
for e in agilex franka; do
    $D4RT -u scripts/probe_flow_layers.py --embodiment $e --refs d4rt --models --group 1 --epochs 60 --output artifacts/downstream/flow_layers_ep60_seed777 $EXTRA >> $L 2>&1
done
conda activate videomae; export LD_PRELOAD=/usr/lib64/libgnutls.so.30
for e in agilex franka; do
    python -u scripts/probe_flow_layers.py --embodiment $e --refs videomaev2 --models --group 1 --epochs 60 --output artifacts/downstream/flow_layers_ep60_seed777 --csv-name layers_videomaev2.csv $EXTRA >> $L 2>&1
done
stage "seed777 pipeline complete"
