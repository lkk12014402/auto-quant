#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! -f "$1" ]]; then
  printf 'Usage: CUDA_VISIBLE_DEVICES=<three-idle-gpus> %s <validated-run.yaml>\n' "$0" >&2
  exit 2
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Set CUDA_VISIBLE_DEVICES to exactly three verified idle physical GPU ids." >&2
  exit 2
fi

python3 - "$CUDA_VISIBLE_DEVICES" <<'PY'
import csv
import subprocess
import sys

selected = [part.strip() for part in sys.argv[1].split(",")]
if len(selected) != 3 or len(set(selected)) != 3 or any(not item.isdigit() for item in selected):
    raise SystemExit("CUDA_VISIBLE_DEVICES must contain exactly three distinct physical GPU indices.")

query = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-gpu=index,uuid,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ],
    text=True,
)
devices = {}
for row in csv.reader(query.splitlines()):
    index, uuid, used, utilization = (field.strip() for field in row)
    devices[index] = (uuid, int(used), int(utilization))

missing = [index for index in selected if index not in devices]
if missing:
    raise SystemExit(f"Unknown physical GPU indices: {', '.join(missing)}")

selected_uuids = set()
for index in selected:
    uuid, used, utilization = devices[index]
    print(f"GPU {index}: {used} MiB used, {utilization}% utilization")
    selected_uuids.add(uuid)
    if used > 100 or utilization > 5:
        raise SystemExit(f"GPU {index} is busy; choose idle cards and retry.")

apps = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ],
    text=True,
)
for row in csv.reader(apps.splitlines()):
    if row and row[0].strip() in selected_uuids:
        raise SystemExit(f"GPU running a compute process: {row[0].strip()}; choose idle cards and retry.")
PY

command -v aqa >/dev/null || {
  echo "aqa is not on PATH; activate the verified AutoRound environment first." >&2
  exit 2
}

export AR_DISABLE_META_LOAD=1
aqa validate "$1"
aqa run "$1" --dry-run
aqa run "$1"
