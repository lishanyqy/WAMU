# OpenWAM RoboTwin Clean Video Training

This README is for running the RoboTwin clean-video experiment on a server or cluster.
Run all commands from the root directory of this repository.

## 1. Prepare the environment

The default environment manager is Conda. It creates an environment named `openwam`
with Python 3.11 or 3.12 and installs the project dependencies.

```bash
bash env_prepare.sh
conda activate openwam
```

If the server uses uv instead:

```bash
bash env_prepare.sh --uv
```

To use a different Conda environment name:

```bash
OPEN_WAM_CONDA_ENV=my-openwam bash env_prepare.sh
conda activate my-openwam
```

## 2. Download the base video model

The experiment uses `Wan2.2-TI2V-5B` by default.

```bash
bash scripts/download_base_models.sh
```

The script downloads the model and updates the matching video-backbone config
automatically. To store the model somewhere else:

```bash
OPEN_WAM_MODEL_ROOT=/path/to/model_storage \
bash scripts/download_base_models.sh
```

## 3. Download the dataset

Download `RoboTwin2.0` using the interactive downloader:

```bash
python scripts/download_assets/download_benchmark_data.py
```

Select `(0) RoboTwin2.0-clean_50` in the menu.

Set the path to the downloaded `dataset` directory:

```bash
export OPEN_WAM_ROBOTWIN_DATA=/path/to/robotwin2.0/dataset
```

The expected directory is normally:

```text
assets/benchmark_data/robotwin2.0/dataset
```

## 4. Training configuration

Use:

```text
configs/train_robotwin_clean_video.yaml
```

Important settings in this config:

- Dataset variant: `clean_50`
- Three-camera input: head, left, and right
- Video loss enabled: `lambda_video: 1.0`
- Action loss disabled: `lambda_action: 0.0`
- Action backbone frozen
- Mixed precision: `bf16`
- Default per-GPU batch size: `8`
- Default gradient accumulation: `4`

## 5. Start training

Single GPU:

```bash
torchrun --nproc_per_node=1 scripts/train.py \
  --config-name train_robotwin_clean_video
```

Four GPUs:

```bash
torchrun --nproc_per_node=4 scripts/train.py \
  --config-name train_robotwin_clean_video
```

`--nproc_per_node` should match the number of GPUs allocated to the job.
Training outputs are written to:

```text
outputs/robotwin_clean_video/
```

## 6. Useful overrides

Run a short debug test before a full job:

```bash
torchrun --nproc_per_node=1 scripts/train.py \
  --config-name train_robotwin_clean_video \
  training.debug=true
```

Override the dataset path without exporting an environment variable:

```bash
torchrun --nproc_per_node=1 scripts/train.py \
  --config-name train_robotwin_clean_video \
  dataloader.dataset_dir=/path/to/robotwin2.0/dataset
```

Reduce memory usage by changing the batch size and accumulation steps:

```bash
torchrun --nproc_per_node=4 scripts/train.py \
  --config-name train_robotwin_clean_video \
  training.batch_size=1 \
  training.gradient_accumulation_steps=32
```

If the GPUs do not support bf16, use fp16:

```bash
torchrun --nproc_per_node=1 scripts/train.py \
  --config-name train_robotwin_clean_video \
  training.mixed_precision=fp16
```

Before submitting a long job, confirm the repository and data path:

```bash
pwd
git status --short --branch
echo "$OPEN_WAM_ROBOTWIN_DATA"
```
