#!/bin/bash
# Run the DINOv2 dense reference only after the VideoMAEv2 one, so at most two dense jobs hold RAM at once.
while screen -ls | grep -q "\.dense_vmae"; do sleep 300; done
echo "$(date) VideoMAEv2 job finished; starting DINOv2 dense reference"
bash /home/a1905721/workspace/OpenWAM/run/probe_dense_layers.sh dinov2 agilex franka
