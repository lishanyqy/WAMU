#!/bin/bash
# Single serial GPU queue on the login node for wamu/action_read_subspace_plan.md (replaces local_worker_v2.sh x2 and
# stitch_chain.sh, which competed for the shared GPU).  One job of ours on the GPU at a time:
#   1. a PENDING (not held, dependencies met) ar_{probe,causal,dump,3b}_* Slurm job: hold it, run it here, then drop it
#      from downstream dependencies and cancel the Slurm copy (on failure: release it back and back off);
#   2. otherwise the next step-5 stitching step (smoke, video M1, M2, M3), state in artifacts/action_read/stitch_state;
#   3. otherwise, while the contact chain is still running (it submits the contact probes at its end), wait.
source ~/.bashrc >/dev/null 2>&1; module load CUDAcompat/12.2-535.161.08 2>/dev/null
cd /home/a1905721/workspace/OpenWAM
export USE_TF=0 PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
LOG=artifacts/logs/action_read/local_worker.log; ST=artifacts/action_read/stitch_state; touch $ST
PY=/home/a1905721/.conda/envs/fastwam/bin/python
freegb() { nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk -F, '{print int(($2-$1)/1024)}'; }
freeram() { free -g | awk '/^Mem:/ {print $7}'; }
ours_on_gpu() { pgrep -u a1905721 -f "fastwam/bin/python -u scripts/action_read/(probes|read_subspace|stitch|step3_bcd)" > /dev/null; }
drop_dep() {   # remove job $1 from every downstream dependency
    local J=$1
    for D in $(squeue -u a1905721 -h -o "%i %E" | awk -v j=$J 'index($2, j) {print $1}'); do
        dep=$(squeue -j $D -h -o "%E" | sed -E "s/\(unfulfilled\)//g; s/\(failed\)//g" | sed -E "s/(afterok|afterany):?//g" | tr ':,' '\n\n' | grep -v "^$J" | grep -E '^[0-9_]+' | paste -sd:)
        typ=$(squeue -j $D -h -o "%E" | grep -oE 'afterok|afterany' | head -1)
        if [ -n "$dep" ]; then scontrol update jobid=$D dependency=$typ:$dep; else scontrol update jobid=$D dependency=""; fi
        echo "$(date)   $D dependency -> ${dep:-none}" >> $LOG
    done
}
echo "$(date) local_queue start" >> $LOG
while true; do
    while ours_on_gpu; do sleep 60; done
    J=$(squeue -u a1905721 -h -t PD -o "%i %j %r" | awk '$2 ~ /^ar_(fit|causal|dump|probe|3b)_/ && $3 !~ /Dependency|Held|JobHeld/ {print $1}' | sort -n | head -1)
    if [ -n "$J" ]; then
        until [ "$(freegb)" -ge 18 ]; do sleep 60; done
        r=$(squeue -j $J -h -o "%T %r" 2>/dev/null); case "$r" in PENDING*Held*|"") continue;; esac
        mkdir -p artifacts/action_read/locks; mkdir artifacts/action_read/locks/$J 2>/dev/null || continue
        scontrol hold $J 2>/dev/null || continue
        name=$(scontrol show job $J | grep -oP 'JobName=\K\S+'); cmd=$(scontrol show job $J | grep -oP '^\s*Command=\K.*')
        echo "$(date) local run of $J $name: $cmd" >> $LOG
        bash $cmd > artifacts/logs/action_read/local_${name}_$J.log 2>&1; rc=$?
        if [ $rc -eq 0 ]; then drop_dep $J; scancel $J; echo "$(date) $J $name done locally" >> $LOG
        else scontrol release $J; rmdir artifacts/action_read/locks/$J; echo "$(date) $J $name FAILED locally (rc $rc); released" >> $LOG; sleep 300; fi
        continue
    fi
    next=""; for s in smoke M1 M2 M3; do grep -qx "$s" $ST || { next=$s; break; }; done
    if [ -n "$next" ]; then
        until [ "$(freegb)" -ge 22 ] && [ "$(freeram)" -ge 100 ]; do sleep 60; done
        echo "$(date) stitch $next" >> $LOG
        if [ $next = smoke ]; then
            $PY -u scripts/action_read/stitch.py --video-model M1 --limit-windows 2 --train-batch 2 --val-windows 2 --lrs 1e-4 > artifacts/logs/action_read/stitch_smoke.log 2>&1; rc=$?
            [ $rc -eq 0 ] && mv artifacts/action_read/results/step5/step5_stitch_videoM1.csv artifacts/action_read/results/step5/smoke_step5_stitch_videoM1.csv
        else
            $PY -u scripts/action_read/stitch.py --video-model $next > artifacts/logs/action_read/stitch_$next.log 2>&1; rc=$?
        fi
        if [ $rc -eq 0 ]; then echo $next >> $ST; echo "$(date) stitch $next done locally" >> $LOG
        else echo "$(date) stitch $next FAILED locally (rc $rc)" >> $LOG; sleep 600; fi
        continue
    fi
    if screen -ls | grep -q ar_contact; then sleep 300; continue; fi
    echo "$(date) local_queue: nothing left; exit" >> $LOG; exit 0
done
