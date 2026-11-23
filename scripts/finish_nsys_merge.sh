#!/usr/bin/env bash
# Finish window_manifest + merge-capture when docker left profile/ root-owned.
# Run from the host with a root-capable container that can write into RUN_DIR.
set -euo pipefail

RUN_DIR="${1:?usage: finish_nsys_merge.sh <run_dir>}"
IP_ROOT="${INFINIPROFILER_ROOT:-/opt/infiniprofiler}"
IMAGE="${FINISH_MERGE_IMAGE:-vllm/vllm-openai:v0.27.1}"

if [[ ! -f "${RUN_DIR}/profile/nsys_truth.json" ]]; then
  echo "missing ${RUN_DIR}/profile/nsys_truth.json" >&2
  exit 1
fi
if [[ ! -f "${RUN_DIR}/client/measured.json" ]]; then
  echo "missing ${RUN_DIR}/client/measured.json" >&2
  exit 1
fi

ROLE="${ROLE:-baseline}"
WINDOW_ID="${WINDOW_ID:-${ROLE}-window}"
CAMPAIGN_ID="${CAMPAIGN_ID:-qwen3-8b-a100-nsys-matchup}"
SERVED_MODEL="${SERVED_MODEL:-qwen3-8b}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
PROFILES_SOURCE_HASH="${PROFILES_SOURCE_HASH:-vllm-openai-v0.27.1}"
BASELINE_PIN="${BASELINE_PIN:-${PROFILES_SOURCE_HASH}}"
CANDIDATE_PIN="${CANDIDATE_PIN:-${PROFILES_SOURCE_HASH}}"
WALL_WORKLOAD_ID="${WALL_WORKLOAD_ID:-qwen3-8b-mixed-256}"
CONCURRENCY="${CONCURRENCY:-16}"
PROMPT_LENGTHS="${PROMPT_LENGTHS:-256,512,1024}"
OUTPUT_LENGTH="${OUTPUT_LENGTH:-256}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"

docker run --rm --entrypoint bash \
  -e ROLE -e WINDOW_ID -e CAMPAIGN_ID -e SERVED_MODEL \
  -e TENSOR_PARALLEL_SIZE -e PROFILES_SOURCE_HASH \
  -e BASELINE_PIN -e CANDIDATE_PIN -e WALL_WORKLOAD_ID \
  -e CONCURRENCY -e PROMPT_LENGTHS -e OUTPUT_LENGTH -e MAX_NUM_SEQS \
  -e PYTHONPATH=/opt/infiniprofiler/src \
  -v "${IP_ROOT}:/opt/infiniprofiler:ro" \
  -v "${RUN_DIR}:/work" \
  "${IMAGE}" \
  -lc '
set -euo pipefail
python3 - /work <<'"'"'PY'"'"'
import hashlib, json, os, sys
from pathlib import Path
work = Path(sys.argv[1])
truth = json.loads((work / "profile/nsys_truth.json").read_text(encoding="utf-8"))
client = json.loads((work / "client/measured.json").read_text(encoding="utf-8"))
rank_pids = {str(rank["rank"]): str(rank["pid_key"]) for rank in truth["ranks"]}
spec = {
    "schema_version": "infiniprofiler-capture-spec-1",
    "role": os.environ["ROLE"],
    "window_id": os.environ["WINDOW_ID"],
    "campaign_id": os.environ["CAMPAIGN_ID"],
    "model": os.environ["SERVED_MODEL"],
    "baseline_pin": os.environ.get("BASELINE_PIN", os.environ["PROFILES_SOURCE_HASH"]),
    "candidate_pin": os.environ.get("CANDIDATE_PIN", os.environ["PROFILES_SOURCE_HASH"]),
    "tensor_parallel": int(os.environ["TENSOR_PARALLEL_SIZE"]),
    "wall_workload_id": os.environ.get("WALL_WORKLOAD_ID", "qwen3-8b-mixed-256"),
    "wall_compare_ref": "/work/client/measured.json",
    "warmup_excluded": True,
    "client_command": "vllm_fixed_client measured window",
    "concurrency": int(os.environ["CONCURRENCY"]),
    "prompt_lengths": [int(item) for item in os.environ["PROMPT_LENGTHS"].split(",")],
    "output_lengths": [int(os.environ["OUTPUT_LENGTH"])],
    "batch_size": int(os.environ.get("MAX_NUM_SEQS", "16")),
    "clock_policy": "operator_observed",
    "profiles_source_hash": os.environ["PROFILES_SOURCE_HASH"],
    "rank_pids": rank_pids,
}
spec["client_command_sha256"] = hashlib.sha256(spec["client_command"].encode()).hexdigest()
rank_rows = []
for row in truth["ranks"]:
    rank = int(row["rank"])
    trace = work / "profile" / f"rank={rank:02d}" / "raw" / "tracer_out-0.json"
    rank_rows.append({
        "rank": rank,
        "pid": int(row["pid_key"]),
        "status": "SUCCEEDED",
        "raw_files": [str(trace)],
        "raw_bytes": trace.stat().st_size,
    })
manifest = {
    "schema_version": "infiniprofiler-capture-manifest-1",
    "status": "SUCCEEDED",
    "started_epoch_ns": int(client["start_epoch_ns"]),
    "ended_epoch_ns": int(client["end_epoch_ns"]),
    "duration_ns": int(client["wall_duration_ns"]),
    "client_returncode": 0,
    "detail": "Nsight session start/stop bounded measured client window",
    "spec": spec,
    "ranks": rank_rows,
}
(work / "profile/capture_spec.json").write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
(work / "profile/window_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print("manifest_ok", len(rank_rows))
PY
python3 -m infiniprofiler merge-capture --capture-dir /work/profile
'

echo "finished merge for ${RUN_DIR}"
