#!/bin/bash
# Run the remaining stitch jobs one after another on this login node's GPU, each only if Slurm has not started it.
cd /home/a1905721/workspace/OpenWAM
for J in 16390227 16390230 16390229; do
    st=$(squeue -j $J -h -o "%T" 2>/dev/null)
    [ "$st" = "PENDING" ] || continue
    bash run/action_read/run_held_job.sh $J
done
