# Download WAMU real

The dataset `lishanyang/wamu_real` was confirmed private on 2026-11-11.
The script checks current visibility and pins each download to the resolved commit.

Use an OpenWAM Python environment, or install the downloader dependency:

```bash
python3 -m pip install -U huggingface_hub
hf auth login
bash download_wamureal.sh
```

Each collaborator needs their own Hugging Face account and read access to the
private dataset. The owner must arrange repository access (for example through
an organization with the appropriate membership), or make the dataset public
if sharing it publicly is intended. Login alone does not grant access.

Use each collaborator's own read token. Never embed an owner's token in this
script, commit it, or distribute it with the dataset. `HF_TOKEN` is supported
for automated jobs; inject it using the job's secret mechanism. The script also
accepts the active credential saved by `hf auth login`.

Default destination: `datasets/lishanyang_wamu_real` under this repository.
An explicit destination is interpreted relative to the invoking shell:

```bash
bash download_wamureal.sh /path/to/datasets/wamu_real
OPEN_WAM_PYTHON=/path/to/env/bin/python bash download_wamureal.sh
```

The complete snapshot includes `bowls_lerobot`, `cupstack_lerobot`,
`eggpack_lerobot`, and `headphones_lerobot` (approximately 16 GB at the verified
revision). Rerun the same command after an interruption to resume. Optional
`WAMUREAL_MAX_WORKERS` controls download concurrency (default: 8).
