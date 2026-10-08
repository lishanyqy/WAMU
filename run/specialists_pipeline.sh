#!/bin/bash
# After the SAM and DA3 extractions and the early-started probe screens finish: remaining probes, zero-shot ceilings, report rebuild.
cd /home/a1905721/workspace/OpenWAM
L=artifacts/logs/specialists_pipeline.log
while screen -ls | grep -qE "\.(sam_extract|da3_extract|da3_depth|probe_da3)\b"; do sleep 120; done
echo "$(date) extractions + DA3 probes finished" >> $L
bash run/probe_dense_layers.sh da3 agilex franka >> $L 2>&1          # no-op if probe_da3 completed everything
while screen -ls | grep -qE "\.probe_sam\b"; do sleep 120; done        # SAM Agilex probes (early screen) must not share the CSV
bash run/probe_dense_layers.sh sam agilex franka >> $L 2>&1
/home/a1905721/.conda/envs/d4rt/bin/python -u scripts/collect_zeroshot.py >> $L 2>&1
/home/a1905721/.conda/envs/d4rt/bin/python scripts/build_probe_report.py >> $L 2>&1
/home/a1905721/.conda/envs/d4rt/bin/python scripts/build_probe_pdf.py >> $L 2>&1
echo "$(date) specialists pipeline complete" >> $L
