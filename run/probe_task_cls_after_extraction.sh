#!/bin/bash
# Waits until OpenWAM extraction (screen gt_openwam) and the current probe run (gt_probe_cls) have both
# finished, then reruns the per-layer probes so models extracted later (mutual, video_sees_action) are
# probed too. Finished (model, layer) rows are skipped, so only the missing ones are computed.
while screen -ls | grep -qE "\.gt_openwam|\.gt_probe_cls|\.gt_probe_mutual"; do sleep 300; done
echo "$(date) extraction and first probe pass finished; probing remaining models"
bash /home/a1905721/workspace/OpenWAM/run/probe_task_cls_layers.sh
echo "$(date) done"
