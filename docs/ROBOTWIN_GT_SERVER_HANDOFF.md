# OpenWAM Robotwin GT 提取服务器交接说明

## 目标

请在这台服务器上运行 Robotwin 的物理轨迹回放，提取与 OpenWAM 已有 feature 对齐的 ground-truth。当前只需要生成 GT，不需要重新提取 OpenWAM、DINOv2、DepthV3 或其它 feature。

每个 episode 最多保存前 132 个 replay frame。已有 temporal feature 使用 `num_frames=33`、`frame_stride=4`，最终对齐帧为 `0, 4, 8, ..., 128`。

配置只启用 `head_camera`，关闭 wrist/front camera。达到 132 帧后应立即停止物理回放，而不是等待机器人完成整条轨迹。

## 明确的 GT 范围

不是导出 Robotwin 的所有完整视频和所有相机。需要处理两个 manifest 中的全部 episode：

```text
Agilex: 2500 episodes（50 tasks × 50 episodes）
Franka: 2300 episodes（46 tasks × 50 episodes）
```

每个 episode 只采集 `head_camera` 的前 132 帧，并保留以下 GT：

```text
observation/head_camera/rgb                 # 用于帧对齐校验
observation/head_camera/depth               # Depth probing
observation/head_camera/actor_segmentation  # 目标/物体监督
observation/head_camera/mesh_segmentation   # mesh/object 监督
observation/head_camera/intrinsic_cv        # 3D 几何投影
observation/head_camera/extrinsic_cv        # 3D 几何投影
observation/head_camera/cam2world_gl        # 3D 几何投影
pointcloud                                  # 3D/flow probing
endpose                                     # 机器人状态
joint_action                                # 机器人关节状态
```

生成 HDF5 后，主服务器会从每个 episode 选择与 feature 完全对应的 33 帧：
`0, 4, 8, ..., 128`。因此对方不需要重新提取任何 OpenWAM 或基础模型
feature，也不需要导出 wrist/front camera、完整 episode 视频或额外相机。

## 需要下载的 Hugging Face 仓库

必须下载：

```text
lishanyang/openwam-robotwin-runtime
lishanyang/openwam-robotwin-source
```

GT 生成后上传回：

```text
lishanyang/openwam-robotwin-gt
```

当前阶段不需要下载：

```text
lishanyang/openwam-robotwin-features
```

feature 已经在主分析服务器上，不需要复制到 GT 提取服务器。

## 下载

```bash
hf download lishanyang/openwam-robotwin-runtime \
  --repo-type dataset \
  --local-dir "$HOME/openwam-robotwin-runtime"

hf download lishanyang/openwam-robotwin-source \
  --repo-type dataset \
  --local-dir "$HOME/openwam-robotwin-source"
```

另外 clone OpenWAM 主仓库；当前 GT 提取不需要运行 OpenWAM 模型：

```bash
git clone https://github.com/OpenWAM-Official/OpenWAM.git \
  "$HOME/workspace/OpenWAM"
```

## Source 数据结构

```text
$HOME/openwam-robotwin-source/data/agilex/<task>/aloha-agilex_clean_50/
$HOME/openwam-robotwin-source/data/franka/<task>/franka_clean_50/
```

每个 clean50 目录只包含 `_traj_data/*.pkl`、`seed.txt` 和 `scene_info.json`。原始 RGB 视频不在 source 仓库中，也不是 replay GT 提取的必需品。

## 安装 Robotwin runtime

先准备官方 RoboTwin 环境（环境名建议为 `RoboTwin`），然后：

```bash
export ROBOTWIN_ROOT="$HOME/workspace/RoboTwin"
bash "$HOME/openwam-robotwin-runtime/setup_robotwin.sh" "$ROBOTWIN_ROOT"
```

setup 脚本会在目标目录不存在时 clone RoboTwin，并安装 GT 提取所需的 patched files、配置和 early-stop 逻辑。

## 准备 manifests

```bash
mkdir -p "$HOME/workspace/OpenWAM/artifacts"
cp "$HOME/openwam-robotwin-source/manifests/"*.jsonl \
  "$HOME/workspace/OpenWAM/artifacts/"
```

## 设置路径

```text
AGILEX_SRC=$HOME/openwam-robotwin-source/data/agilex
FRANKA_SRC=$HOME/openwam-robotwin-source/data/franka
```

运行前修改 runtime 中的 batch script，以及以下两个 YAML 配置中的路径：

```text
RoboTwin/task_config/gt_export.yml
RoboTwin/task_config/gt_export_franka.yml
```

关键配置必须保持：

```yaml
camera:
  collect_head_camera: true
  collect_wrist_camera: false
  collect_front_camera: false
max_gt_frames: 132
stop_after_gt_frames: true
```

`save_path` 改成可写目录，例如 `$HOME/robotwin_gt/agilex` 和 `$HOME/robotwin_gt/franka`。

## 运行

```bash
cd "$HOME/openwam-robotwin-runtime"
bash scripts/run_robotwin_gt_export.example.sh agilex
bash scripts/run_robotwin_gt_export.example.sh franka
```

可以分别运行两个 embodiment，也可以在两台独立服务器上并行运行。Robotwin 物理 replay 主要使用 CPU；GPU 不占用不代表 GT 提取失败。已有的 `data/episode*.hdf5` 会被脚本跳过，因此可以中断后续跑。

## 输出检查

每个完成的 HDF5 应至少包含：

```text
observation/head_camera/rgb
observation/head_camera/depth
observation/head_camera/actor_segmentation
observation/head_camera/mesh_segmentation
observation/head_camera/intrinsic_cv
observation/head_camera/extrinsic_cv
pointcloud
endpose
joint_action
```

长 episode 的帧数应为 132；短 episode 可以少于 132。GT 提取器会对短 episode 使用 linspace 选择 33 帧。

## 回传 GT

```bash
hf upload lishanyang/openwam-robotwin-gt \
  /path/to/robotwin_gt \
  . \
  --repo-type dataset \
  --commit-message "Add Robotwin Agilex and Franka GT"
```

回传时请同时提供：source/runtime 版本、Agilex 和 Franka 完成的 episode 数量、失败任务和 episode 列表、HDF5 文件数量，以及长 episode 是否均为 132 帧。主分析服务器会用已有 temporal features 和这些 GT 做下游 probing。
