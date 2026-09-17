import os
from huggingface_hub import snapshot_download

repo_id = "depth-anything/DA3-BASE"
local_dir = "/home/a1905721/modelspace/depthanything"

snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    local_dir=local_dir,
    # token=os.getenv("HF_TOKEN"),  # 公共模型可删除这一行
    resume_download=True,
    max_workers=8,
)

print(f"模型已下载到：{local_dir}")