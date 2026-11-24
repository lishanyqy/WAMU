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



### deploy

Deploy the WAMU real self-attention checkpoint at step 20000 using the
[OpenWAM deployment guide](assets/openwam_usage_docs/train-and-deploy.md).
On a fresh checkout with Conda installed, run `bash env_prepare.sh --conda`
from the OpenWAM repo once to create the `openwam` environment.

```bash
cd /home/King/WAMU_real/WAMU
conda activate openwam

checkpoint_subdir=action_film/wamu_real_self_attn/2026-11-21_12-56-52
checkpoint_file=checkpoint_step_20000.safetensors

# Skip this download if the checkpoint bundle is already under ../models.
hf download paperupload/WAMs --repo-type dataset --local-dir ../models \
  --include "$checkpoint_subdir/$checkpoint_file" \
  --include "$checkpoint_subdir/config.yaml" \
  --include "$checkpoint_subdir/normalization_stats.npy" \
  --include "$checkpoint_subdir/tokenizer/*"

bash scripts/deploy.sh "../models/$checkpoint_subdir" \
  --ckpt-name "$checkpoint_file" \
  --device cuda:0 --host 127.0.0.1 --port 8848 --compile-enabled false
```

The server runs in the foreground at `ws://127.0.0.1:8848`. Once it reports
that it is listening, verify the connection from another terminal.

```bash
conda activate openwam
python - <<'PY'
import json
from websockets.sync.client import connect

with connect("ws://127.0.0.1:8848", open_timeout=5) as ws:
    ws.send(json.dumps({"type": "ping"}))
    response = json.loads(ws.recv(timeout=5))
    assert response["type"] == "pong", response
    print(response)
PY
```



#### Run the client

From another terminal on the server machine, run the single-inference smoke
test with synthetic camera frames and a 14-value zero state.

```bash
cd /home/King/WAMU_real/WAMU
conda activate openwam

python scripts/inference_test/inference_single_test.py \
  --server ws://127.0.0.1:8848 \
  --test --state-dim 14
```

To run one inference with recorded observations, use three camera frames and
the joint state from the same observation.

```bash
python scripts/inference_test/inference_single_test.py \
  --server ws://127.0.0.1:8848 \
  --head-camera /path/to/top.png \
  --left-wrist-camera /path/to/left_wrist.png \
  --right-wrist-camera /path/to/right_wrist.png \
  --state-file /path/to/state.json \
  --prompt "Pick up the cup."
```

Both commands print the inference result. For a client on another machine,
run `ssh -N -L 8848:127.0.0.1:8848 user@gpu-host` on the client machine and
keep the tunnel open while connecting to `ws://127.0.0.1:8848`.

#### Inference inputs

The server accepts JSON messages with `type: "obs"`, `"ping"`, or `"reset"`.
For this checkpoint, an `obs` message takes the following fields.

- `images.head_camera` is the required top camera frame. The two wrist frames
go in `images.left_wrist_camera` and `images.right_wrist_camera`. Each image
is a base64-encoded PNG or JPEG on the wire. Omitted or null wrist frames
are filled with black images.
- `state` is required and contains 14 raw joint-state values in training order.
For the CLI, `state.json` can be a JSON array of 14 numbers or an object
with a `state` array. The server normalizes and maps this input to 80 dimensions.
- `prompt` is the task instruction string. The server uses an empty string
if it is omitted.

Each action response contains a 14-value joint action in the original
training-data scale after server-side inverse normalization.