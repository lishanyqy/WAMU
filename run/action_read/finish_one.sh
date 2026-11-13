#!/bin/bash
# Finish a locally running job whose worker loop was stopped: wait for PID, then do what local_worker.sh would have done.
cd /home/a1905721/workspace/OpenWAM; PID=$1; J=$2; OUT=$3; LOG=artifacts/logs/action_read/local_worker.log
while kill -0 $PID 2>/dev/null; do sleep 30; done
if [ -f "$OUT" ] && [ "$OUT" -nt artifacts/logs/action_read/local_worker.log.start_$J ]; then
    for D in $(squeue -u a1905721 -h -o "%i %E" | awk -v j=$J 'index($2, j) {print $1}'); do
        dep=$(squeue -j $D -h -o "%E" | sed -E "s/\(unfulfilled\)//g; s/\(failed\)//g" | sed -E "s/(afterok|afterany):?//g" | tr ':,' '\n\n' | grep -v "^$J" | grep -E '^[0-9_]+' | paste -sd:)
        typ=$(squeue -j $D -h -o "%E" | grep -oE 'afterok|afterany' | head -1)
        if [ -n "$dep" ]; then scontrol update jobid=$D dependency=$typ:$dep; else scontrol update jobid=$D dependency=""; fi
        echo "$(date)   $D dependency -> ${dep:-none}" >> $LOG
    done
    scancel $J; echo "$(date) $J done locally (finisher)" >> $LOG
else
    scontrol release $J; rmdir artifacts/action_read/locks/$J 2>/dev/null; echo "$(date) $J FAILED locally (finisher); released to Slurm" >> $LOG
fi
