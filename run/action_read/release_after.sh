#!/bin/bash
# Release held Slurm jobs once the given jobs have been claimed by a local worker (lock dir exists) or have left the queue.
# usage: release_after.sh "<jobs to wait for>" "<jobs to release>"
cd /home/a1905721/workspace/OpenWAM
WAIT=$1; REL=$2
for j in $WAIT; do
    until [ -d artifacts/action_read/locks/$j ] || [ -z "$(squeue -j $j -h 2>/dev/null)" ]; do sleep 60; done
done
for j in $REL; do scontrol release $j; done
echo "$(date) released $REL after $WAIT started" >> artifacts/logs/action_read/local_worker.log
