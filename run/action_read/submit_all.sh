#!/bin/bash
# Submit the whole wamu/action_read_subspace_plan.md pipeline to Slurm as a dependency graph:
#   fit (step 1, 3 models x ID/OOD) -> causal (step 0/2: same-set R; OOD with ID's R for 4d; M1 ID twice for the
#   determinism check) -> stage_step3.sh (analysis + acceptance gate; submits step 3 only if k* exists)
#   step 5 stitching (one job per video model) runs independently.
#   token GT is (re)built after the contact replay array finishes (afterany: partial failures are flagged per window).
set -eu
cd /home/a1905721/workspace/OpenWAM
J=run/action_read/job.sbatch; C=run/action_read/cpu.sbatch
CONTACT_JOB=${CONTACT_JOB:-16362866}
mkdir -p artifacts/logs/action_read; : > artifacts/action_read/dump_jobs.txt
declare -A FIT
for m in M1 M2 M3; do for s in ID OOD; do
    FIT[$m$s]=$(sbatch --parsable --time=06:00:00 --job-name=ar_fit_${m}_$s $J read_subspace.py --model $m --set $s --mode fit)
done; done
CAUSAL=()
for m in M1 M2 M3; do
    for s in ID OOD; do
        CAUSAL+=($(sbatch --parsable --time=12:00:00 --dependency=afterok:${FIT[$m$s]} --job-name=ar_causal_${m}_$s $J read_subspace.py --model $m --set $s --mode causal --layer-windows 8 32 128))
    done
    CAUSAL+=($(sbatch --parsable --time=12:00:00 --dependency=afterok:${FIT[${m}ID]} --job-name=ar_causal_${m}_OOD_RID $J read_subspace.py --model $m --set OOD --r-from ID --mode causal))
done
CAUSAL+=($(sbatch --parsable --time=12:00:00 --dependency=afterok:${FIT[M1ID]} --job-name=ar_causal_M1_ID_rerun $J read_subspace.py --model M1 --set ID --mode causal --layer-windows 8 32 128 --tag rerun))
# token GT after the contact replay
GT=()
for s in ID OOD; do GT+=($(sbatch --parsable --dependency=afterany:$CONTACT_JOB --job-name=ar_gt_$s $C build_token_gt.py --set $s --workers 32)); done
printf '%s\n' "${GT[@]}" > artifacts/action_read/gt_jobs.txt
DEP=$(IFS=:; echo "${CAUSAL[*]}")
STAGE=$(sbatch --parsable --dependency=afterok:$DEP --job-name=ar_stage3 --wrap="bash run/action_read/stage_step3.sh" --account=acvt --partition=icelake --cpus-per-task=4 --mem=32G --time=02:00:00 --output=artifacts/logs/action_read/ar_stage3_%j.log)
# step 5: stitching, one job per video model (3 models on CPU + 1 on GPU -> more host memory)
ST=()
for m in M1 M2 M3; do ST+=($(sbatch --parsable --mem=300G --time=24:00:00 --job-name=ar_stitch_$m $J stitch.py --video-model $m)); done
echo "fit: ${FIT[*]}"; echo "causal: ${CAUSAL[*]}"; echo "gt: ${GT[*]}"; echo "stage3: $STAGE"; echo "stitch: ${ST[*]}"
