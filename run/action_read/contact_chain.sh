#!/bin/bash
# Finish the contact replay on the login node (the Slurm array was not being scheduled), then rebuild the token GT with
# the complete contact labels and submit contact-only probes (picked up by the local workers).
# Login node: its driver is old (11.6) -> CUDAcompat MUST be loaded (on compute nodes it must be stripped).  Loaded ONCE
# here: sourcing ~/.bashrc inside the loop clobbers the loop variable.
source ~/.bashrc >/dev/null 2>&1; module load CUDAcompat/12.2-535.161.08 2>/dev/null
cd /home/a1905721/workspace/OpenWAM
LOG=artifacts/logs/action_read/contact_chain.log
freegb() { nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk -F, '{print int(($2-$1)/1024)}'; }
echo "$(date) contact chain start (v3)" >> $LOG
for IDX in 1 2 3 4 5 6 7 8 9; do
    until [ "$(freegb)" -ge 8 ]; do sleep 120; done
    echo "$(date) contact replay array index $IDX" >> $LOG
    (cd /tmp && HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 /home/a1905721/.conda/envs/RoboTwin/bin/python -I \
        /home/a1905721/dataspace/openwam-robotwin-contact/tools/run_contact.py --workers 2 --timeout 1800 --array-index $IDX --array-count 10) \
        >> /home/a1905721/dataspace/openwam-robotwin-contact/logs/local_index_$IDX.log 2>&1
    echo "$(date)   index $IDX exit $?" >> $LOG
done
n=$(find /home/a1905721/dataspace/openwam-robotwin-contact/agilex /home/a1905721/dataspace/openwam-robotwin-contact/agilex_random -name "*_contact.hdf5" | wc -l)
echo "$(date) contact files: $n / 1000" >> $LOG
[ "$n" -ge 990 ] || { echo "$(date) contact replay incomplete ($n); NOT rebuilding GT / submitting probes" >> $LOG; exit 1; }
for s in ID OOD; do
    OMP_NUM_THREADS=1 /home/a1905721/.conda/envs/fastwam/bin/python -u scripts/action_read/build_token_gt.py --set $s --workers 16 >> $LOG 2>&1
done
for m in M1 M2 M3; do for spec in "ID ID" "OOD OOD" "ID OOD"; do set -- $spec
    sbatch --parsable --time=03:00:00 --job-name=ar_probe_contact_${m}_$1to$2 run/action_read/job.sbatch probes.py --model $m --train-set $1 --test-set $2 --k 32 --only-contents contact >> $LOG
done; done
echo "$(date) contact chain done" >> $LOG
