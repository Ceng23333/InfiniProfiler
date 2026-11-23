# Roadmap

## M0 - Timing-first normalized abstraction (done)

- `infiniprofiler-normalize-1` / `infiniprofiler-profile-2` timing-attribution
  contracts (profile-1 smoke retired).
- hcTracer/mcTracer Chrome-trace normalization with truncated capture recovery.
- Streaming exclusive timeline attribution (`api_submit_us`, `gpu_execution_us`,
  `sync_wait_us`, graph launch/replay, CPU gaps).
- Nsight SQLite adapter mapping CUPTI truth into the same timing buckets.
- Metrics-only sidecar parser/producer and `/proc` fallback sampler.
- AutoHelix request emission with `stop_before_promote=true`.

Promotion-grade matched-window, rank, source-map, and mechanism work is
specified in [p0-p1-plan.md](p0-p1-plan.md). Ranking uses exclusive timing
ratios and optional engine metrics — never `infinilm:` scheduler span contracts.

## M1 - A100 Nsight-matched capture (done / continuing)

1. Matched vLLM TP4 Qwen3-8B BF16 windows on host `A100` (GPUs 0–3).
2. `compare_nsys_gate.py` PASS on timing families only.
3. Harden capture env export / root-owned merge finish path.
4. Emit profile-2 + AutoHelix request from matched nsys merges.

Exit: closed L1 timing loop on A100 without scheduler-span dependence.

## M2 - Engine metrics sidecar (optional L2)

Emit `tokens_per_iteration`, `batch_size`, and `queue_depth` at the matched
window boundary. Sidecar spans may appear for provenance (L3) but are not
promotion evidence; packing/queue paths are metric-only hints.

Exit: baseline and candidate sidecars share window identity and metric names.

## M3 - Optional engine ranges (L3 provenance)

Prototype NVTX/HCTX/sidecar ranges around host loops. Ingest into
provenance/warnings only. A future `engine.span_*` rank path requires matched
A/B ranges and explicit thresholds — still never inferred from CUDA gaps.

Exit: ranges visible in artifacts; ranker does not invent `scheduler.*`.

## M4 - Matched A/B + AutoHelix campaign

Use the unprofiled promotion wall first. Invoke nsys/hcTracer/sidecar when the
wall regresses. Feed one baseline and one candidate profile into
`promotion-request`, then write the IO-compatible compare/backlog.

Exit: `runtime.*` timing paths and `engine.*` metric hints remain separate in
the ranked backlog. AutoHelix consumes the request with `stop_before_promote`.

## M5 - Qwen3-8B InfiniLM promotion gate

Promotion is reportable only when the unprofiled primary wall ratio is greater
than the operator threshold, and all wall guardrails pass. A profiled timing
improvement alone is a diagnosis, not a promotion.

## Acceptance gate (NVIDIA A100)

Matched Nsight Systems vs InfiniProfiler TP4 compare on host `A100` using
`scripts/compare_nsys_gate.py`. PASS requires timing-family thresholds only —
no dependence on `scheduler_signals` / `infinilm:` spans. Showroom mcTracer is
optional smoke, not the gate.
