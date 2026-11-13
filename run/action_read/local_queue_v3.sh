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
    J=$(squeue -u a1905721 -h -t PD -o "%i %j %r" | awk '$2 ~ /^ar_(fit|causal|dump|probe|3b|stitch)_/ && $3 !~ /Dependency|Held|JobHeld/ {print $1}' | sort -n | head -1)
    if [ -n "$J" ]; then
        case "$(squeue -j $J -h -o %j)" in ar_stitch_*) needg=22; needr=100;; *) needg=18; needr=0;; esac
        until [ "$(freegb)" -ge $needg ] && [ "$(freeram)" -ge $needr ]; do sleep 60; done
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
    # (step-5 stitching is now submitted as Slurm jobs ar_stitch_* and picked up above like every other job)
    if [ "$(squeue -u a1905721 -h -o %j | grep -c '^ar_')" -gt 0 ]; then sleep 120; continue; fi
    echo "$(date) local_queue: nothing left; exit" >> $LOG; exit 0
done
