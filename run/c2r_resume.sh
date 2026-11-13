#!/bin/bash
# Resume the clean2rand probes (finished layers are skipped by the probe scripts).
cd /home/a1905721/workspace/OpenWAM
for w in ${@:-flow cls}; do
  bash run/probe_c2r.sh $w >> artifacts/logs/c2r_probe_$w.log 2>&1
  echo "$(date '+%F %T') probes $w exit=$?" >> artifacts/logs/c2r_pipeline.log
done
echo "$(date '+%F %T') ALL DONE" >> artifacts/logs/c2r_pipeline.log
