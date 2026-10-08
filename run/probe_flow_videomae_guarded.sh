#!/bin/bash
# The node is shared; the VideoMAEv2 flow job (~36 GB peak) has been OOM-killed by neighbours twice. Run it only when
# >= 70 GB is available and re-queue after a kill; the per-layer CSV makes each attempt resume where it stopped.
cd /home/a1905721/workspace/OpenWAM
for attempt in $(seq 1 20); do
    until [ "$(free -g | awk 'NR==2{print $7}')" -ge 70 ]; do sleep 120; done
    echo "$(date) attempt $attempt, $(free -g | awk 'NR==2{print $7}') GB available"
    bash run/probe_flow_layers.sh videomaev2 agilex franka
    n=$(cat artifacts/downstream/flow_layers_ep60/*/layers_videomaev2.csv 2>/dev/null | grep -c videomaev2)
    [ "$n" -ge 48 ] && { echo "$(date) VideoMAEv2 flow complete ($n rows)"; exit 0; }
    echo "$(date) exited with $n/48 rows; waiting before retry"; sleep 300
done
