#!/bin/bash
# Submit step 3 for the given models with a common k (default 32): probes (same set + ID->OOD), 3b, 3c, 3d, each after
# that model's dump job.  Dump job ids are read from artifacts/action_read/dump_jobs_<model>.txt (lines: <set> <jobid>).
# usage: bash run/action_read/submit_step3.sh 32 M1 M2
cd /home/a1905721/workspace/OpenWAM
K=$1; shift
J=run/action_read/job.sbatch
/home/a1905721/.conda/envs/fastwam/bin/python scripts/action_read/pick_3d_windows.py
for m in "$@"; do
    dID=$(awk '$1=="ID" {print $2}' artifacts/action_read/dump_jobs_$m.txt)
    dOOD=$(awk '$1=="OOD" {print $2}' artifacts/action_read/dump_jobs_$m.txt)
    for s in ID OOD; do
        if [ $s = ID ]; then d=$dID; else d=$dOOD; fi
        sbatch --parsable --dependency=afterok:$d --time=08:00:00 --job-name=ar_probe_${m}_$s $J probes.py --model $m --train-set $s --k $K
        sbatch --parsable --dependency=afterok:$d --time=04:00:00 --job-name=ar_3b_${m}_$s $J step3_bcd.py --part b --model $m --set $s --k $K
        sbatch --parsable --dependency=afterok:$d --time=02:00:00 --job-name=ar_probe_3c_${m}_$s $J step3_bcd.py --part c --model $m --set $s
        sbatch --parsable --dependency=afterok:$d --time=02:00:00 --job-name=ar_probe_3d_${m}_$s $J step3_bcd.py --part d --model $m --set $s --k $K --layer 19
    done
    sbatch --parsable --dependency=afterok:$dID:$dOOD --time=08:00:00 --job-name=ar_probe_${m}_ID2OOD $J probes.py --model $m --train-set ID --test-set OOD --k $K
done
