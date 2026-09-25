#SDK模型下载
from modelscope import snapshot_download
model_dir = snapshot_download(
    'gotobn8/4D_fastwam_final',
    token="ms-ed06d124-a437-4ae5-bd92-a0914da9a1e9"
)

# /data/user/prtroas0003/lishan/modelspace/ckpt/fastwam/robotwin_uncond_3cam_384_1e-4_test/2026-07-22_12-52-40/checkpoints/state

# /data/user/prtroas0003/lishan/modelspace/ckpt/fastwam/robotwin_uncond_3cam_384_1e-4_test/2026-07-22_12-52-40/checkpoints/state/step_024000/

# robotwin_uncond_3cam_384_1e-4_test