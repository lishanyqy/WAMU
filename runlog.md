## Run Log for WAMU

### 2026-09-26 (v0.0)

```bash
# Activate your environment
conda activate openwam
# export your robotwin data env
export OPEN_WAM_ROBOTWIN_DATA=/path/to/robotwin2.0/dataset
torchrun --nproc_per_node=4 scripts/train.py \
  --config-name train_robotwin_clean_action_film.yaml\
```