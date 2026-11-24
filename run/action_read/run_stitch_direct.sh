#!/bin/bash
# Run one stitch video model directly on this login node (no Slurm copy left for it), logging like the other runs.
M=$1; cd /home/a1905721/workspace/OpenWAM; LOG=artifacts/logs/action_read/local_worker.log
echo "$(date) $(hostname) direct run of stitch video $M (no Slurm job)" >> $LOG
bash run/action_read/job.sbatch stitch.py --video-model $M > artifacts/logs/action_read/local_ar_stitch_${M}_direct.log 2>&1; rc=$?
echo "$(date) $(hostname) stitch video $M direct run exit $rc" >> $LOG
