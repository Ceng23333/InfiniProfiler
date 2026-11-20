#!/usr/bin/env bash
set -euo pipefail

# Deterministic Observe/Parse helper. It intentionally exits after emitting the
# AutoHelix request; lever selection and rebuild remain outside this script.
PYTHON_BIN="${PYTHON_BIN:-python3}"
SOURCE="${SOURCE:-hctracer}"
INPUT="${INPUT:?INPUT is the raw hcTracer JSON or naive-sidecar JSONL}"
PROFILE="${PROFILE:-profile/profile.json}"
WALL_COMPARE="${WALL_COMPARE:?WALL_COMPARE is required}"
REQUEST="${REQUEST:-profile/autohelix_request.json}"
BASELINE="${BASELINE:-}"
CAMPAIGN_ID="${CAMPAIGN_ID:?CAMPAIGN_ID is required}"
BASELINE_PIN="${BASELINE_PIN:?BASELINE_PIN is required}"
CANDIDATE_PIN="${CANDIDATE_PIN:?CANDIDATE_PIN is required}"
ITERATION="${ITERATION:?ITERATION is required}"
MODEL="${MODEL:-Qwen3-8B}"
HOST="${HOST:-metax-9-sg}"
DEVICE_TRACK="${DEVICE_TRACK:-mars}"
EXHAUSTED="${EXHAUSTED:-}"

normalize_args=(normalize --source "${SOURCE}" --input "${INPUT}" --output "${PROFILE}")
if [[ -n "${BASELINE}" ]]; then
  normalize_args+=(--baseline "${BASELINE}")
fi

"${PYTHON_BIN}" -m infiniprofiler "${normalize_args[@]}"
"${PYTHON_BIN}" -m infiniprofiler promotion-request \
  --profile "${PROFILE}" \
  --wall-compare "${WALL_COMPARE}" \
  --output "${REQUEST}" \
  --campaign-id "${CAMPAIGN_ID}" \
  --model "${MODEL}" \
  --host "${HOST}" \
  --device-track "${DEVICE_TRACK}" \
  --baseline-pin "${BASELINE_PIN}" \
  --candidate-pin "${CANDIDATE_PIN}" \
  --iteration "${ITERATION}" \
  --exhausted "${EXHAUSTED}"

echo "infiniprofiler measure complete: ${REQUEST}"
echo "stop_before_promote=true"
