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

### 2026-11-11 (v0.2)
Concat Action FiLM
```bash
# Activate your environment
conda activate openwam
# export your robotwin data env
export OPEN_WAM_ROBOTWIN_DATA=/path/to/robotwin2.0/dataset
torchrun --nproc_per_node=4 scripts/train.py \
  --config-name train_robotwin_clean_concat_action_film.yaml \
  model.video_backbone.model_path=assets/video_backbone_ckpt/Wan2.2-TI2V-5B \
  training.output_path=/scratch/ou29/rz6525/WAMU/WAMU/outputs/self_attn_clean \
```

### 2026-10-08 (v1.0)
cross attention action sees video
```bash
# Activate your environment
conda activate openwam
# export your robotwin data env
export OPEN_WAM_ROBOTWIN_DATA=/path/to/robotwin2.0/dataset
torchrun --nproc_per_node=4 scripts/train.py \
  --config-name train_robotwin_clean_cross_attn.yaml \
  model.video_backbone.model_path=assets/video_backbone_ckpt/Wan2.2-TI2V-5B \
  training.output_path=/scratch/ou29/rz6525/WAMU/WAMU/outputs/cross_attn_clean \
```

### 2026-10-15 (v1.1)
cross attention action only sees video but detach the loss conduction.
```bash
# Activate your environment
conda activate openwam
# export your robotwin data env
export OPEN_WAM_ROBOTWIN_DATA=/path/to/robotwin2.0/dataset
torchrun --nproc_per_node=4 scripts/train.py \
  --config-name train_robotwin_clean_cross_attn_detach.yaml \
  model.video_backbone.model_path=assets/video_backbone_ckpt/Wan2.2-TI2V-5B \
  training.output_path=/scratch/ou29/rz6525/WAMU/WAMU/outputs/cross_attn_detach_robotwin_clean \
  # training.debug=true
```

### 2026-11-01 (v1.2)
joint softmax for action and video
```bash
# Activate your environment
conda activate openwam
# export your robotwin data env
export OPEN_WAM_ROBOTWIN_DATA=/path/to/robotwin2.0/dataset
torchrun --nproc_per_node=4 scripts/train.py \
  --config-name train_robotwin_clean_cross_attention_softmax.yaml \
  model.video_backbone.model_path=assets/video_backbone_ckpt/Wan2.2-TI2V-5B \
  training.output_path=/scratch/ou29/rz6525/WAMU/WAMU/outputs/cross_attention_softmax
```


### 2026-11-02 (v1.3)
joint-self attn for video detached from action
```bash
# Activate your environment
conda activate openwam
# export your robotwin data env
export OPEN_WAM_ROBOTWIN_DATA=/path/to/robotwin2.0/dataset
torchrun --nproc_per_node=4 scripts/train.py \
  --config-name train_robotwin_clean_self_attn_detach_video.yaml \
  model.video_backbone.model_path=assets/video_backbone_ckpt/Wan2.2-TI2V-5B \
  training.output_path=/scratch/ou29/rz6525/WAMU/WAMU/outputs/self-attn-detach
```

### 2026-11-03 (v1.4)
bridge noise (wamu/bridge_noise_plan.md): during training only, corrupt the video features the ACTION stream reads
(70% of samples, 10-20% of video tokens in 2x2 blocks, Gaussian or token swap, same tokens in every layer, same plan for
both architectures). Cross-attention: the bridge features; self-attention: the video K/V of the action rows (video rows
and the video loss stay clean). Each run keeps its baseline recipe; the only change is `bridge_noise.enabled: true`.
```bash
# Activate your environment
conda activate openwam
# export your robotwin data env
export OPEN_WAM_ROBOTWIN_DATA=/path/to/robotwin2.0/dataset
# cross-attention + bridge noise (recipe of cross_attn_clean: 1024 bridge, grad accum 4)
torchrun --nproc_per_node=4 scripts/train.py \
  --config-name train_robotwin_clean_cross_attn_bridge_noise.yaml \
  model.video_backbone.model_path=assets/video_backbone_ckpt/Wan2.2-TI2V-5B \
  training.output_path=/scratch/ou29/rz6525/WAMU/WAMU/outputs/cross_attn_bridge_noise
```
# self-attention + bridge noise (recipe of self_attention_30k / sft_from_scratch_robotwin_clean: grad accum 1)

### 2026-11-06 (v1.5)
Which cross attention video token is using projections from video.
```bash
# Activate your environment
conda activate openwam
# export your robotwin data env
export OPEN_WAM_ROBOTWIN_DATA=/path/to/robotwin2.0/dataset
# cross-attention + direct projection
torchrun --nproc_per_node=4 scripts/train.py \
  --config-name train_robotwin_clean_cross_attention_direct.yaml \
  model.video_backbone.model_path=assets/video_backbone_ckpt/Wan2.2-TI2V-5B \
  training.output_path=/scratch/ou29/rz6525/WAMU/WAMU/outputs/cross_attn_direct
```

### 2026-10-22 (v2.0) 
bug fix for v1.0, they have a apparent bug that might impact on the the cross-attention architecture.
```bash
# Activate your environment
conda activate openwam
# export your robotwin data env
export OPEN_WAM_ROBOTWIN_DATA=/path/to/robotwin2.0/dataset
torchrun --nproc_per_node=4 scripts/train.py \
  --config-name train_robotwin_clean_cross_attn_fix.yaml \
  model.video_backbone.model_path=assets/video_backbone_ckpt/Wan2.2-TI2V-5B \
  training.output_path=/scratch/ou29/rz6525/WAMU/WAMU/outputs/cross_attn_clean_fix3072 \
```
### 2026-11-13 (v3.0)
WAMU real (WAMsU/wamu_real: bowls / cupstack / eggpack / headphones), joint self-attention from Wan2.2 only (no robot pretraining).
action_sees_video, no FiLM, shared quantile stats (`<data>/meta/stats.json`), 20k steps.
```bash
# Activate your environment
conda activate openwam
# export your WAMU real data env (default: datasets/lishanyang_wamu_real)
export OPEN_WAM_WAMU_REAL_DATA=/path/to/wamu_real
torchrun --nproc_per_node=4 scripts/train.py \
  --config-name train_wamu_real_self_attn.yaml \
  model.video_backbone.model_path=assets/video_backbone_ckpt/Wan2.2-TI2V-5B \
  training.output_path=/scratch/ou29/rz6525/WAMU/WAMU/outputs/wamu_real_self_attn
```

### 2026-11-13 (v3.1)
WAMU real, joint self-attention warm-started from the robot-only pretrained OpenWAM/pretrain_robot_only.
Same recipe as v3.0 (the checkpoint is also action_sees_video).

**Please download OpenWAM/pretrain_robot_only**
```bash
# Activate your environment
conda activate openwam
# export your WAMU real data env (default: datasets/lishanyang_wamu_real)
export OPEN_WAM_WAMU_REAL_DATA=/path/to/wamu_real
torchrun --nproc_per_node=4 scripts/train.py \
  --config-name train_wamu_real_pretrain_robot_self_attn.yaml \
  model.video_backbone.model_path=assets/video_backbone_ckpt/Wan2.2-TI2V-5B \
  training.finetune_ckpt_path=/path/to/pretrain_robot_only \
  training.output_path=/scratch/ou29/rz6525/WAMU/WAMU/outputs/wamu_real_pretrain_robot_self_attn
```

### 2026-11-13 (v3.2)
WAMU real, warm-started from the ego + robot pretrained OpenWAM/OpenWAM-Alpha-Pretrain-Foundation-Model.
Same recipe as v3.0 except the attention mask keeps the checkpoint's own `action sees video`.

**Please download OpenWAM/pretrain_ego_robot_cotrain_action_sees_video**
```bash
# Activate your environment
conda activate openwam
# export your WAMU real data env (default: datasets/lishanyang_wamu_real)
export OPEN_WAM_WAMU_REAL_DATA=/path/to/wamu_real

torchrun --nproc_per_node=4 scripts/train.py \
  --config-name train_wamu_real_pretrain_foundation.yaml \
  model.video_backbone.model_path=assets/video_backbone_ckpt/Wan2.2-TI2V-5B \
  training.finetune_ckpt_path=/path/to/OpenWAM-Pretrain-Foundation-Model \
  training.output_path=/scratch/ou29/rz6525/WAMU/WAMU/outputs/wamu_real_pretrain_foundation
```

### 2026-11-25 (v3.3)
WAMU real, **single task** (`bowls`), warm-started from OpenWAM/pretrain_robot_only.
Global batch 128 (16 per GPU x 4 GPUs x grad accum 2), 20k steps. Gripper slots same as v3.0-v3.2
(`unify_action_map: ["0-6", "34-40"]`, inherited from v3.0).
Pick another task with `dataloader.tasks=[cupstack]` / `[eggpack]` / `[headphones]`.

**Please download OpenWAM/pretrain_robot_only**
```bash
# Activate your environment
conda activate openwam
# export your WAMU real data env (default: datasets/lishanyang_wamu_real)
export OPEN_WAM_WAMU_REAL_DATA=/path/to/wamu_real
torchrun --nproc_per_node=4 scripts/train.py \
  --config-name train_wamu_real_pretrain_robot_single_task.yaml \
  model.video_backbone.model_path=assets/video_backbone_ckpt/Wan2.2-TI2V-5B \
  training.finetune_ckpt_path=/path/to/pretrain_robot_only \
  'dataloader.tasks=[bowls]' \
  training.output_path=/scratch/ou29/rz6525/WAMU/WAMU/outputs/wamu_real_pretrain_robot_single_task_bowls
```

### 2026-11-25 (v3.4)
续训任务，ckpt从之前训练完的https://huggingface.co/datasets/paperupload/WAMs/blob/main/action_film/wamu_real_pretrain_foundation/2026-11-21_14-15-41/checkpoint_step_20000.safetensors 这个checkpoint开始续训. 之前有保存目录的话，直接training.finetune_ckpt_path指向上级目录就行了，这个是四个任务的，数据集跟v3.0,v3.1 v3.2是一致的
Same recipe as v3.2 (gripper slots `["0-6", "34-40"]`, global batch 256, 20k steps) except `learning_rate: 3e-5` (v3.2: 1e-4).

**Please download the whole run folder paperupload/WAMs: action_film/wamu_real_pretrain_foundation/2026-11-21_14-15-41/** (config.yaml + tokenizer/ + normalization_stats.npy + checkpoint_step_*.safetensors)
```bash
# Activate your environment
conda activate openwam
# export your WAMU real data env (default: datasets/lishanyang_wamu_real)
export OPEN_WAM_WAMU_REAL_DATA=/path/to/wamu_real
# finetune_ckpt_path: the run folder (highest step = 20000 is picked) or .../checkpoint_step_20000.safetensors
torchrun --nproc_per_node=4 scripts/train.py \
  --config-name train_wamu_real_pretrain_cotrain_multi_task.yaml \
  model.video_backbone.model_path=assets/video_backbone_ckpt/Wan2.2-TI2V-5B \
  training.finetune_ckpt_path=/path/to/wamu_real_pretrain_foundation/2026-11-21_14-15-41 \
  training.output_path=/scratch/ou29/rz6525/WAMU/WAMU/outputs/wamu_real_pretrain_cotrain_multi_task
```

### 2026-11-25 (v3.5)
WAMU real, **single task** (`bowls`), warm-started from the human (ego) + robot co-trained
OpenWAM/pretrain_ego_robot_cotrain_action_sees_video. 
.
这个用的就是从huggingface上下载下来的ckpt，repo_id = OpenWAM/pretrain_ego_robot_cotrain_action_sees_video
这个是只训一个任务。
**Please download OpenWAM/pretrain_ego_robot_cotrain_action_sees_video**
```bash
# Activate your environment
conda activate openwam
# export your WAMU real data env (default: datasets/lishanyang_wamu_real)
export OPEN_WAM_WAMU_REAL_DATA=/path/to/wamu_real
torchrun --nproc_per_node=4 scripts/train.py \
  --config-name train_wamu_real_pretrain_cotrain_single_task.yaml \
  model.video_backbone.model_path=assets/video_backbone_ckpt/Wan2.2-TI2V-5B \
  training.finetune_ckpt_path=/path/to/pretrain_ego_robot_cotrain_action_sees_video \
  'dataloader.tasks=[bowls]' \
  training.output_path=/path/to/WAMU/outputs/wamu_real_pretrain_cotrain_single_task_bowls
```

### 2026-11-26 (v3.6)
续训任务：从训完的 v3.0 (joint self-attention, Wan2.2 only) 的 step 20000 续训，四个任务，数据集跟 v3.0/v3.1/v3.2 一致。
Same recipe as v3.4 (gripper slots `["0-6", "34-40"]`, global batch 256, 20k steps, `learning_rate: 3e-5`).

**Please download the whole run folder paperupload/WAMs: action_film/wamu_real_self_attn/2026-11-21_12-56-52/** (config.yaml + tokenizer/ + normalization_stats.npy + checkpoint_step_*.safetensors)
```bash
# Activate your environment
conda activate openwam
# export your WAMU real data env (default: datasets/lishanyang_wamu_real)
export OPEN_WAM_WAMU_REAL_DATA=/path/to/wamu_real
# finetune_ckpt_path: the run folder (highest step = 20000 is picked) or .../checkpoint_step_20000.safetensors
torchrun --nproc_per_node=4 scripts/train.py \
  --config-name train_wamu_real_self_attn_continue.yaml \
  model.video_backbone.model_path=assets/video_backbone_ckpt/Wan2.2-TI2V-5B \
  training.finetune_ckpt_path=/path/to/wamu_real_self_attn/2026-11-21_12-56-52 \
  training.output_path=/scratch/ou29/rz6525/WAMU/WAMU/outputs/wamu_real_self_attn_continue
```

### 2026-11-26 (v3.7)
续训任务：从训完的 v3.1 (warm-started from OpenWAM/pretrain_robot_only) 的 step 20000 续训，四个任务，数据集跟 v3.0/v3.1/v3.2 一致。
Same recipe as v3.4 (gripper slots `["0-6", "34-40"]`, global batch 256, 20k steps, `learning_rate: 3e-5`).

**Please download the whole run folder paperupload/WAMs: action_film/wamu_real_pretrain_robot_self_attn/2026-11-21_14-57-31/** (config.yaml + tokenizer/ + normalization_stats.npy + checkpoint_step_*.safetensors)
```bash
# Activate your environment
conda activate openwam
# export your WAMU real data env (default: datasets/lishanyang_wamu_real)
export OPEN_WAM_WAMU_REAL_DATA=/path/to/wamu_real
# finetune_ckpt_path: the run folder (highest step = 20000 is picked) or .../checkpoint_step_20000.safetensors
torchrun --nproc_per_node=4 scripts/train.py \
  --config-name train_wamu_real_pretrain_robot_self_attn_continue.yaml \
  model.video_backbone.model_path=assets/video_backbone_ckpt/Wan2.2-TI2V-5B \
  training.finetune_ckpt_path=/path/to/wamu_real_pretrain_robot_self_attn/2026-11-21_14-57-31 \
  training.output_path=/scratch/ou29/rz6525/WAMU/WAMU/outputs/wamu_real_pretrain_robot_self_attn_continue
```
