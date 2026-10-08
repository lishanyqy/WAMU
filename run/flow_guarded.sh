#!/bin/bash
# Memory-guarded, self-resuming runner for the flow/tracking sweeps on a crowded shared node.
# Usage: bash run/flow_guarded.sh <openwam|videomaev2|dinov2> <min_free_GB> <expected_rows_per_csv> <emb...>
# Waits until >= min_free_GB is available, runs the sweep, and re-queues after an OOM kill; each attempt resumes
# from the per-layer CSVs, so nothing finished is redone.
cd /home/a1905721/workspace/OpenWAM
what=$1; minfree=$2; want=$3; shift 3; embs="$@"
csv=layers.csv; [ "$what" = videomaev2 ] && csv=layers_videomaev2.csv; [ "$what" = dinov2 ] && csv=layers_dinov2.csv
count() { t=0; for e in $embs; do f=artifacts/downstream/flow_layers_ep60/$e/$csv; [ -f $f ] && t=$((t + $(wc -l < $f) - 1)); done; echo $t; }
for attempt in $(seq 1 40); do
    [ "$(count)" -ge "$want" ] && { echo "$(date) $what complete ($(count)/$want rows)"; exit 0; }
    until [ "$(free -g | awk 'NR==2{print $7}')" -ge "$minfree" ]; do sleep 120; done
    echo "$(date) $what attempt $attempt ($(count)/$want rows done, $(free -g | awk 'NR==2{print $7}') GB free)"
    GROUP=1 bash run/probe_flow_layers.sh $what $embs
    sleep 180
done
echo "$(date) $what gave up after 40 attempts ($(count)/$want rows)"
