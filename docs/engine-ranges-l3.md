# Optional L3 engine ranges (provenance only)

Scheduler / admission / batch effort is **not** recoverable from CUPTI idle gaps.
When an engine can emit named host ranges (NVTX, HCTX, or sidecar JSONL spans),
InfiniProfiler may ingest them as **L3 provenance**.

## Rules

1. L1 Nsight timing remains the promotion diagnosis primary.
2. Span rows are aggregated into `capture.provenance.engine_ranges`.
3. Spans never revive `scheduler_signals` / InfiniLM canonical marker contracts.
4. Ranker paths stay in `runtime.*` / `graph.*` / `engine.*` (metric hints) /
   `diagnosis.insufficient_evidence`. Gaps do not become packing.
5. A future `engine.span_*` rank path needs matched A/B ranges, shared
   `window_id`, and explicit ratio thresholds — not implemented as a contract yet.

## Prototype

```bash
PYTHONPATH=src python3 scripts/engine_range_proto.py \
  --out-jsonl /tmp/engine-ranges.jsonl \
  --out-report /tmp/engine-ranges-report.json
```

Attach to a matched A100 profile-2 build:

```bash
PYTHONPATH=src python3 scripts/build_a100_profile2_request.py \
  --baseline <baseline_run> \
  --candidate <candidate_run> \
  --engine-ranges-jsonl /tmp/engine-ranges.jsonl \
  --out-dir <out_dir>
```
