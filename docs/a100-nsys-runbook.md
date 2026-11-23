# A100 Nsight Systems matchup runbook

Acceptance host: `ssh A100` (use GPUs 0–3 only; leave 4–7 alone).

## Pins

| Pin | Value |
|---|---|
| Image | `vllm/vllm-openai:v0.27.1` |
| Model | `/data/shared/wuhang/models/Qwen3-8B` → `/models/qwen3-8b` |
| Devices | GPUs `0-3`, TP4 |
| Nsight | host `2025.3.2` bind-mounted to `/opt/nsight-systems` |
| Workload | 16 warmup + 48 measured, concurrency 16, prompts `256/512/1024`, output 256 |
| Remote root | `/home/zenghua/workspace/infiniprofiler-vs-nsys/` |

## Capture

```bash
# Inside the container (see scripts/vllm_tp_nsys_capture.sh)
# All manifest env vars are exported by the script, including MAX_NUM_SEQS.
```

If merge fails because `profile/` is root-owned after docker exit:

```bash
export INFINIPROFILER_ROOT=/home/zenghua/workspace/infiniprofiler-vs-nsys/InfiniProfiler
export ROLE=baseline WINDOW_ID=baseline-window-...
bash "${INFINIPROFILER_ROOT}/scripts/finish_nsys_merge.sh" \
  /home/zenghua/workspace/infiniprofiler-vs-nsys/runs/<run_dir>
```

## Gate

```bash
PYTHONPATH=src python3 scripts/compare_nsys_gate.py \
  --baseline runs/nsys-... \
  --candidate runs/infiniprofiler-... \
  --expected-ranks 4 \
  --output comparison.json
```

PASS requires timing-family thresholds only — no `scheduler_signals`.

## Profile-2 + AutoHelix

```bash
PYTHONPATH=src python3 scripts/build_a100_profile2_request.py \
  --baseline <baseline_run> \
  --candidate <candidate_run> \
  --out-dir <out_dir>
```

Emits `profile-v2.json` and `autohelix-request.json` with `stop_before_promote=true`.

Optional L3 engine-range provenance (never `scheduler_signals`):

```bash
PYTHONPATH=src python3 scripts/engine_range_proto.py \
  --out-jsonl /tmp/engine-ranges.jsonl \
  --out-report /tmp/engine-ranges-report.json

PYTHONPATH=src python3 scripts/build_a100_profile2_request.py \
  --baseline <baseline_run> \
  --candidate <candidate_run> \
  --engine-ranges-jsonl /tmp/engine-ranges.jsonl \
  --out-dir <out_dir>
```

See `docs/engine-ranges-l3.md`.
