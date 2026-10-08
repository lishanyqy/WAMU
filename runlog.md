## Run Log for WAMU

### 2026-09-26 (v0.0)

```bash
# Activate your environment
conda activate openwam
# export your robotwin data env
export OPEN_WAM_ROBOTWIN_DATA=/path/to/robotwin2.0/dataset
torchrun --nproc_per_node=4 scripts/train.py \
  --config-name train_robotwin_clean_action_film.yaml \
  model.video_backbone.model_path=assets/video_backbone_ckpt/Wan2.2-TI2V-5B \
  training.output_path=/scratch/ou29/rz6525/WAMU/WAMU/outputs/mean_action_film \
```

### 2026-09-29 (v0.1)
Concat Action FiLM
```bash
# Activate your environment
conda activate openwam
# export your robotwin data env
export OPEN_WAM_ROBOTWIN_DATA=/path/to/robotwin2.0/dataset
torchrun --nproc_per_node=4 scripts/train.py \
  --config-name train_robotwin_clean_concat_action_film.yaml \
  model.video_backbone.model_path=assets/video_backbone_ckpt/Wan2.2-TI2V-5B \
  training.output_path=/scratch/ou29/rz6525/WAMU/WAMU/outputs/concat_action_film \
```


### 2026-10-08 (v1.0)
Concat Action FiLM
```bash
# Activate your environment
conda activate openwam
# export your robotwin data env
export OPEN_WAM_ROBOTWIN_DATA=/path/to/robotwin2.0/dataset
torchrun --nproc_per_node=4 scripts/train.py \
  --config-name train_robotwin_clean_cross_attn.yaml.yaml \
  model.video_backbone.model_path=assets/video_backbone_ckpt/Wan2.2-TI2V-5B \
  training.output_path=/scratch/ou29/rz6525/WAMU/WAMU/outputs/concat_action_film \
```