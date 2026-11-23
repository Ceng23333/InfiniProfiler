#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:?RUN_ID is required}"
ROLE="${ROLE:-candidate}"
CAMPAIGN_ID="${CAMPAIGN_ID:-qwen3-8b-a100-nsys-matchup}"
WINDOW_ID="${WINDOW_ID:-${RUN_ID}}"
MODEL_PATH="${MODEL_PATH:-/models/qwen3-8b}"
SERVED_MODEL="${SERVED_MODEL:-qwen3-8b}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
PORT="${PORT:-8000}"
ENDPOINT="${ENDPOINT:-http://127.0.0.1:${PORT}}"
NSYS_BIN="${NSYS_BIN:-/opt/nsight-systems/target-linux-x64/nsys}"
INFINIPROFILER_ROOT="${INFINIPROFILER_ROOT:-/opt/infiniprofiler}"
WORK_DIR="${WORK_DIR:-/work}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.50}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-16}"
MEASURED_REQUESTS="${MEASURED_REQUESTS:-48}"
CONCURRENCY="${CONCURRENCY:-16}"
PROMPT_LENGTHS="${PROMPT_LENGTHS:-256,512,1024}"
OUTPUT_LENGTH="${OUTPUT_LENGTH:-256}"
SEED="${SEED:-42}"
PROFILES_SOURCE_HASH="${PROFILES_SOURCE_HASH:-vllm-openai-v0.27.1}"
BASELINE_PIN="${BASELINE_PIN:-${PROFILES_SOURCE_HASH}}"
CANDIDATE_PIN="${CANDIDATE_PIN:-${PROFILES_SOURCE_HASH}}"
WALL_WORKLOAD_ID="${WALL_WORKLOAD_ID:-qwen3-8b-mixed-256}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-900}"

# Export every value the inlined Python manifest builder reads from os.environ.
export ROLE WINDOW_ID CAMPAIGN_ID SERVED_MODEL TENSOR_PARALLEL_SIZE
export CONCURRENCY PROMPT_LENGTHS OUTPUT_LENGTH MAX_NUM_SEQS
export PROFILES_SOURCE_HASH BASELINE_PIN CANDIDATE_PIN WALL_WORKLOAD_ID
export WORK_DIR INFINIPROFILER_ROOT

mkdir -p "${WORK_DIR}/nsys" "${WORK_DIR}/client" "${WORK_DIR}/profile"
SESSION="infiniprofiler_${RUN_ID}"
REPORT_BASE="${WORK_DIR}/nsys/profile"
VLLM_LOG="${WORK_DIR}/vllm.log"
EXPORT_LOG="${WORK_DIR}/nsys/export.log"

echo "run_id=${RUN_ID} role=${ROLE} session=${SESSION}" | tee "${WORK_DIR}/capture.log"
set +e
"${NSYS_BIN}" sessions list >"${WORK_DIR}/nsys/sessions-before.txt" 2>&1
set -e

"${NSYS_BIN}" launch \
  --session="${SESSION}" \
  --trace=cuda,nvtx \
  --cuda-graph-trace=node \
  vllm serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    >"${VLLM_LOG}" 2>&1 &
LAUNCH_PID=$!

cleanup() {
  set +e
  kill -TERM "${LAUNCH_PID}" 2>/dev/null || true
  for _ in $(seq 1 60); do
    kill -0 "${LAUNCH_PID}" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "${LAUNCH_PID}" 2>/dev/null; then
    kill -KILL "${LAUNCH_PID}" 2>/dev/null || true
  fi
  wait "${LAUNCH_PID}" 2>/dev/null || true
  set -e
}
trap cleanup EXIT INT TERM

python3 - "${ENDPOINT}/health" "${READY_TIMEOUT_S}" <<'PY'
import sys
import time
import urllib.request

url, timeout_s = sys.argv[1], float(sys.argv[2])
deadline = time.monotonic() + timeout_s
last_error = "not attempted"
while time.monotonic() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            if response.status == 200:
                print("vLLM health check passed")
                raise SystemExit(0)
            last_error = f"HTTP {response.status}"
    except Exception as exc:
        last_error = f"{type(exc).__name__}: {exc}"
    time.sleep(2)
raise SystemExit(f"vLLM did not become ready in {timeout_s}s: {last_error}")
PY

python3 "${INFINIPROFILER_ROOT}/scripts/vllm_fixed_client.py" \
  --endpoint "${ENDPOINT}" \
  --model "${SERVED_MODEL}" \
  --model-path "${MODEL_PATH}" \
  --requests "${WARMUP_REQUESTS}" \
  --concurrency "${CONCURRENCY}" \
  --prompt-lengths "${PROMPT_LENGTHS}" \
  --output-length "${OUTPUT_LENGTH}" \
  --seed 10000 \
  --output "${WORK_DIR}/client/warmup.json"

echo "starting Nsight session ${SESSION}" | tee -a "${WORK_DIR}/capture.log"
"${NSYS_BIN}" start \
  --session="${SESSION}" \
  --sample=none \
  --output="${REPORT_BASE}" \
  --force-overwrite=true \
  --stop-on-exit=true | tee -a "${WORK_DIR}/capture.log"

set +e
python3 "${INFINIPROFILER_ROOT}/scripts/vllm_fixed_client.py" \
  --endpoint "${ENDPOINT}" \
  --model "${SERVED_MODEL}" \
  --model-path "${MODEL_PATH}" \
  --requests "${MEASURED_REQUESTS}" \
  --concurrency "${CONCURRENCY}" \
  --prompt-lengths "${PROMPT_LENGTHS}" \
  --output-length "${OUTPUT_LENGTH}" \
  --seed "${SEED}" \
  --output "${WORK_DIR}/client/measured.json"
CLIENT_RC=$?
set -e

"${NSYS_BIN}" stop --session="${SESSION}" | tee -a "${WORK_DIR}/capture.log"
cleanup
trap - EXIT INT TERM

if [[ "${CLIENT_RC}" -ne 0 ]]; then
  echo "measured client failed rc=${CLIENT_RC}" >&2
  exit "${CLIENT_RC}"
fi

for _ in $(seq 1 120); do
  [[ -f "${REPORT_BASE}.nsys-rep" ]] && break
  sleep 1
done
[[ -f "${REPORT_BASE}.nsys-rep" ]] || { echo "Nsight report was not produced: ${REPORT_BASE}.nsys-rep" >&2; exit 1; }

"${NSYS_BIN}" export \
  --type=sqlite \
  --lazy=false \
  --force-overwrite=true \
  --output="${REPORT_BASE}.sqlite" \
  "${REPORT_BASE}.nsys-rep" >"${EXPORT_LOG}" 2>&1

PYTHONPATH="${INFINIPROFILER_ROOT}/src" python3 - "${REPORT_BASE}.sqlite" "${WORK_DIR}/profile" "${TENSOR_PARALLEL_SIZE}" <<'PY'
import json
import sys
from pathlib import Path

from infiniprofiler.adapters.nsys_sqlite import export_rank_traces, summarize_truth

sqlite_path, output_dir, expected_ranks = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
truth = summarize_truth(sqlite_path, expected_ranks)
output_dir.mkdir(parents=True, exist_ok=True)
(output_dir / "nsys_truth.json").write_text(json.dumps(truth, indent=2, sort_keys=True) + "\n", encoding="utf-8")
result = export_rank_traces(sqlite_path, output_dir, expected_ranks=expected_ranks)
(output_dir / "nsys_conversion.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"rank_count": truth["rank_count"], "ranks": truth["ranks"]}, sort_keys=True))
PY

PYTHONPATH="${INFINIPROFILER_ROOT}/src" python3 - "${WORK_DIR}" <<'PY'
import hashlib
import json
import os
import sys
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
    "wall_compare_ref": str(work / "client/measured.json"),
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
PY

PYTHONPATH="${INFINIPROFILER_ROOT}/src" python3 -m infiniprofiler merge-capture \
  --capture-dir "${WORK_DIR}/profile"

echo "capture complete: ${WORK_DIR}" | tee -a "${WORK_DIR}/capture.log"
