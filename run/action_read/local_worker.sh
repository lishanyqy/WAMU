#!/bin/bash
# Help the Slurm queue with the login-node GPU: repeatedly take one PENDING (not dependency-blocked) action_read GPU job,
# hold it in Slurm, run the same command locally, and on success remove it from downstream dependencies and cancel the
# Slurm copy (on failure: release it back to Slurm).  One job at a time; waits for >= NEED GB of free GPU memory.
cd /home/a1905721/workspace/OpenWAM
NEED=${NEED:-18}; LOG=artifacts/logs/action_read/local_worker.log
freegb() { nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk -F, '{print int(($2-$1)/1024)}'; }
while true; do
    # candidates: our GPU jobs (fit / causal / dump / probe / 3b), pending for Priority/Resources (dependencies met)
    J=$(squeue -u a1905721 -h -t PD -o "%i %j %r" | awk '$2 ~ /^ar_(fit|causal|dump|probe|3b)_/ && $3 !~ /Dependency|Held|JobHeld/ {print $1}' | sort -n | head -1)
    if [ -z "$J" ]; then
        n=$(squeue -u a1905721 -h -o "%j" | grep -cE '^ar_(fit|causal|dump|probe|3b)_')
        [ "$n" -eq 0 ] && { echo "$(date) nothing left; exit" >> $LOG; exit 0; }
        sleep 120; continue
    fi
    until [ "$(freegb)" -ge "$NEED" ]; do sleep 60; done
    scontrol hold $J 2>/dev/null || continue
    name=$(scontrol show job $J | grep -oP 'JobName=\K\S+')
    cmd=$(scontrol show job $J | grep -oP '^\s*Command=\K.*')
    echo "$(date) local run of $J $name: $cmd" >> $LOG
    bash $cmd > artifacts/logs/action_read/local_${name}_$J.log 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then
        for D in $(squeue -u a1905721 -h -o "%i %E" | awk -v j=$J 'index($2, j) {print $1}'); do
            dep=$(squeue -j $D -h -o "%E" | sed -E "s/\(unfulfilled\)//g; s/\(failed\)//g" | sed -E "s/(afterok|afterany):?//g" | tr ':,' '\n\n' | grep -v "^$J" | grep -E '^[0-9_]+' | paste -sd:)
            typ=$(squeue -j $D -h -o "%E" | grep -oE 'afterok|afterany' | head -1)
            if [ -n "$dep" ]; then scontrol update jobid=$D dependency=$typ:$dep; else scontrol update jobid=$D dependency=""; fi
            echo "$(date)   $D dependency -> ${dep:-none}" >> $LOG
        done
        scancel $J; echo "$(date) $J $name done locally" >> $LOG
    else
        scontrol release $J; echo "$(date) $J $name FAILED locally (rc $rc); released to Slurm" >> $LOG
        sleep 300
    fi
done
