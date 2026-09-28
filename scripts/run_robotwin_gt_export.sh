#!/usr/bin/env bash
set -u

ROOT=/home/a1905721/workspace/RoboTwin
AGILEX_SRC=/home/a1905721/dataspace/openwam/benchmark_data/robotwin2.0/dataset
FRANKA_SRC=/home/a1905721/dataspace/robotwin2.0/dataset

run_embodiment() {
    local name="$1" source_root="$2" output_root="$3" config="$4" manifest="$5"
    mapfile -t tasks < <(python - "$manifest" <<'PY'
import json, sys
seen = set()
for line in open(sys.argv[1]):
    task = json.loads(line)["task"]
    if task not in seen:
        seen.add(task)
        print(task)
PY
)
    for task in "${tasks[@]}"; do
        local source="$source_root/$task/${name}_clean_50"
        local target="$output_root/$task/gt_export"
        if [[ ! -d "$source/_traj_data" ]]; then
            echo "[$name] missing source: $source" >&2
            continue
        fi
        mkdir -p "$target"
        if [[ ! -e "$target/_traj_data" ]]; then ln -s "$source/_traj_data" "$target/_traj_data"; fi
        if [[ ! -e "$target/seed.txt" ]]; then ln -s "$source/seed.txt" "$target/seed.txt"; fi
        if [[ -f "$source/scene_info.json" && ! -e "$target/scene_info.json" ]]; then
            ln -s "$source/scene_info.json" "$target/scene_info.json"
        fi
        echo "[$name] start task=$task"
        (cd "$ROOT" && PYTHONUNBUFFERED=1 conda run --no-capture-output -n RoboTwin \
            python script/collect_data.py "$task" "$config") || \
            echo "[$name] FAILED task=$task" >&2
        echo "[$name] done task=$task"
    done
}

case "${1:-all}" in
  agilex) run_embodiment aloha-agilex "$AGILEX_SRC" \
      /home/a1905721/dataspace/openwam/benchmark_data/robotwin2.0/dataset_gt/agilex \
      gt_export "$PWD/artifacts/agilex_robotwin10s_manifest.jsonl" ;;
  franka) run_embodiment franka "$FRANKA_SRC" \
      /home/a1905721/dataspace/robotwin2.0/dataset_gt/franka \
      gt_export_franka "$PWD/artifacts/franka_robotwin10s_manifest.jsonl" ;;
  all) "$0" agilex & "$0" franka & wait ;;
  *) echo "usage: $0 [agilex|franka|all]" >&2; exit 2 ;;
esac
