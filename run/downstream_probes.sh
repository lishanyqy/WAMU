#!/bin/bash
# Downstream probes (wamu/downstreamtask.md) for every OpenWAM model whose extraction is complete.
# Re-running picks up newly extracted models; finished probes and cached features are reused.
# Usage: bash run/downstream_probes.sh <agilex|franka>
source ~/.bashrc
module load CUDAcompat/12.2-535.161.08
conda deactivate; conda activate fastwam
cd /home/a1905721/workspace/OpenWAM
e=$1
n=$(grep -c . artifacts/${e}_robotwin_gt_manifest.jsonl)
ready=()
for d in /home/a1905721/dataspace/openwam_gt_features/${e}/*/; do
    [ "$(ls $d | grep -c '\.npz$')" -eq "$n" ] && ready+=("$(basename $d)")
done
echo "complete models: ${ready[*]}"
python -u scripts/select_layers_cka.py --embodiment $e --models "${ready[@]}"
python -u scripts/run_probes.py --embodiment $e --models "${ready[@]}"
