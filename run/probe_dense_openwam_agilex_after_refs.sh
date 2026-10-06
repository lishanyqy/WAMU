#!/bin/bash
# The four dense jobs together exceed the node's RAM (one was OOM-killed). Start the Agilex OpenWAM sweep
# only after the VideoMAEv2 and DINOv2 reference jobs have finished and freed their ~95 GB.
while screen -ls | grep -qE "\.dense_vmae|\.dense_dino"; do sleep 300; done
echo "$(date) reference jobs finished; starting Agilex OpenWAM dense sweep"
bash /home/a1905721/workspace/OpenWAM/run/probe_dense_layers.sh openwam agilex
