#!/bin/bash
# Run one held Slurm job of ours directly on this (login) node: J = Slurm job id. Lock dir + hold prevent a second copy.
# On success: drop it from downstream dependencies and cancel the Slurm copy; on failure: release it back to Slurm.
J=$1; cd /home/a1905721/workspace/OpenWAM; LOG=artifacts/logs/action_read/local_worker.log
mkdir -p artifacts/action_read/locks; mkdir artifacts/action_read/locks/$J 2>/dev/null || { echo "$(date) $J already locked"; exit 1; }
scontrol hold $J || { rmdir artifacts/action_read/locks/$J; exit 1; }
name=$(scontrol show job $J | grep -oP 'JobName=\K\S+'); cmd=$(scontrol show job $J | grep -oP '^\s*Command=\K.*')
echo "$(date) $(hostname) direct run of $J $name: $cmd" >> $LOG
bash $cmd > artifacts/logs/action_read/local_${name}_$J.log 2>&1; rc=$?
if [ $rc -eq 0 ]; then scancel $J; echo "$(date) $(hostname) $J $name done locally" >> $LOG
else scontrol release $J; rmdir artifacts/action_read/locks/$J; echo "$(date) $(hostname) $J $name FAILED locally (rc $rc); released" >> $LOG; fi
