#!/bin/bash
# Contact GT replay on a LOGIN node (old driver: needs CUDAcompat, unlike the compute nodes). Resumable; 2 processes.
source ~/.bashrc >/dev/null 2>&1
module load CUDAcompat/12.2-535.161.08
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1
cd /tmp
exec /home/a1905721/.conda/envs/RoboTwin/bin/python -I /home/a1905721/dataspace/openwam-robotwin-contact/tools/run_contact.py --workers 2 --timeout 1800 "$@"
