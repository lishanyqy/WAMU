#!/bin/bash
# Runs after every step-1/2 job (Slurm dependency).  1) step 0 + step 2 tables/figures; 2) acceptance gate (plan step 2):
# if every model has a k* at every action sigma on ID, submit step 3 (dump -> probes, 3b, 3c, 3d) per model with its k*;
# otherwise write artifacts/action_read/STOPPED_step2.md and submit nothing (the plan says: report and wait).
set -u
cd /home/a1905721/workspace/OpenWAM
PY=/home/a1905721/.conda/envs/fastwam/bin/python
LOG=artifacts/logs/action_read/stage_step3.log
echo "$(date) stage_step3 start" >> $LOG
$PY scripts/action_read/analyze_causal.py --rerun-a artifacts/action_read/raw/causal_M1_ID_RID.npz --rerun-b artifacts/action_read/raw/causal_M1_ID_RID_rerun.npz >> $LOG 2>&1
# k* per model = max over sigmas on ID with R from ID; empty if not reached at some sigma
KS=$($PY - <<'EOF'
import csv
rows = [r for r in csv.DictReader(open('artifacts/action_read/results/step2_kstar.csv')) if r['set'] == 'ID' and r['R_from'] == 'ID']
out = []
for m in ('M1', 'M2', 'M3'):
    ks = [r['kstar'] for r in rows if r['model'] == m]
    out.append(f"{m}={max(map(int, ks)) if ks and all(ks) else 'NA'}")
print(' '.join(out))
EOF
)
echo "$(date) k* (ID): $KS" >> $LOG
if echo "$KS" | grep -q NA; then
    { echo "# Step 2 acceptance not met"; echo; echo "k* per model (ID, R from ID, max over action sigma): $KS"; echo;
      echo "See results/step2_causal.csv, results/step2_kstar.csv, figs/step2_recovery_curves.pdf. Per the plan, step 3 was NOT started."; } > artifacts/action_read/STOPPED_step2.md
    echo "$(date) acceptance not met; stopping before step 3" >> $LOG; exit 0
fi
# windows for 3d: eval episodes, start 0 (shared by the causal and the step-3 window sets), lowest / highest NMSE_full of M1 at sigma 0.5
$PY - <<'EOF'
import json, numpy as np
for s in ('ID', 'OOD'):
    z = np.load(f'artifacts/action_read/raw/causal_M1_{s}_R{s}.npz', allow_pickle=True)
    v = [str(x) for x in z['variants']]; full = z['nmse'][:, v.index('full')]
    sel = np.isclose(z['action_sigma'], z['action_sigma'][np.argmin(np.abs(z['action_sigma'] - 0.5))]) & np.array([int(k[2]) == 0 for k in z['key']])
    idx = np.nonzero(sel)[0]; o = idx[np.argsort(full[idx])]
    pick = [(i, 'lowest NMSE_full (M1)') for i in o[:3]] + [(i, 'highest NMSE_full (M1)') for i in o[-3:]]
    json.dump([dict(task=str(z['key'][i][0]), episode=int(z['key'][i][1]), start=int(z['key'][i][2]), why=w) for i, w in pick],
              open(f'artifacts/action_read/figs/step3d_windows_{s}.json', 'w'), indent=1)
EOF
GT_DEP=$(cat artifacts/action_read/gt_jobs.txt 2>/dev/null | paste -sd: -)
for MK in $KS; do m=${MK%=*}; k=${MK#*=}
    for s in ID OOD; do
        d=$(sbatch --parsable --job-name=ar_dump_${m}_$s run/action_read/job.sbatch read_subspace.py --model $m --set $s --mode dump)
        dep="afterok:$d${GT_DEP:+:$GT_DEP}"
        sbatch --dependency=$dep --job-name=ar_probe_${m}_$s run/action_read/job.sbatch probes.py --model $m --train-set $s --k $k >> $LOG
        [ $s = OOD ] || true
        sbatch --dependency=$dep --job-name=ar_3b_${m}_$s run/action_read/job.sbatch step3_bcd.py --part b --model $m --set $s --k $k >> $LOG
        sbatch --dependency=$dep --job-name=ar_3c_${m}_$s run/action_read/cpu.sbatch step3_bcd.py --part c --model $m --set $s >> $LOG
        sbatch --dependency=$dep --job-name=ar_3d_${m}_$s run/action_read/cpu.sbatch step3_bcd.py --part d --model $m --set $s --k $k --layer 19 >> $LOG
        echo "$d" >> artifacts/action_read/dump_jobs.txt
    done
    # 4d: probes trained on ID fit, tested on OOD eval (needs both dumps)
    ids=$(tail -n 2 artifacts/action_read/dump_jobs.txt | paste -sd: -)
    sbatch --dependency=afterok:$ids${GT_DEP:+:$GT_DEP} --job-name=ar_probe_${m}_ID2OOD run/action_read/job.sbatch probes.py --model $m --train-set ID --test-set OOD --k $k >> $LOG
done
echo "$(date) step 3 submitted" >> $LOG
# step 4b: novel subspace of M3 (and M2) relative to M1, per set, after all dumps; then probes on it
ALLD=$(paste -sd: artifacts/action_read/dump_jobs.txt)
LAST=""
for s in ID OOD; do for o in M3 M2; do
    kk=$(echo "$KS" | tr ' ' '\n' | grep "^$o=" | cut -d= -f2)
    n=$(sbatch --parsable --dependency=afterok:$ALLD --job-name=ar_4b_${o}vsM1_$s run/action_read/cpu.sbatch step4.py --part novel --base M1 --other $o --set $s)
    p=$(sbatch --parsable --dependency=afterok:$n${GT_DEP:+:$GT_DEP} --job-name=ar_probe_novel_${o}_$s run/action_read/job.sbatch probes.py --model $o --train-set $s --k $kk --novel novel_${o}vsM1_$s.pt)
    LAST="$LAST:$n:$p"
done; done
# 4a summary after everything of step 3/4 (afterany: a failed piece leaves an empty cell, not a missing table)
ALL3=$(squeue -u a1905721 -h -o "%i %j" | awk '$2 ~ /^ar_(dump|probe|3b|3c|3d|4b)_/ {print $1}' | paste -sd:)
sbatch --dependency=afterany:${ALL3} --job-name=ar_summary run/action_read/cpu.sbatch step4.py --part summary >> $LOG
echo "$(date) step 4 submitted" >> $LOG
