import os
from huggingface_hub import snapshot_download

# repo_id = "OpenWAM/robotwin_dual_system_idm"
repo_id = 'OpenGVLab/VideoMAEv2-Large'
# local_dir = "/home/a1905721/dataspace/robotwin2.0"

local_dir = "/home/a1905721/modelspace/VideoMAEv2-Large"

if not os.path.exists(local_dir):
    os.makedirs(local_dir)

snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    local_dir=local_dir,
    # token=os.getenv("HF_TOKEN"),  # 公共模型可删除这一行
    # allow_patterns=["dataset/*/franka_clean_50.zip"],
    resume_download=True,
    max_workers=8,
)

print(f"模型已下载到：{local_dir}")