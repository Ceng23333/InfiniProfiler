# Roadmap

## M0 - Normalized abstraction (done)

- `infiniprofiler-profile-1` in-memory model and JSON schema.
- hcTracer Chrome-trace normalization with truncated capture recovery.
- Explicit HCTX/marker extraction into scheduler signals.
- Dependency-free sidecar parser/producer and `/proc` fallback sampler.
- AutoHelix `promotion-request-1` with `stop_before_promote=true`.

## M1 - metax-9-sg capture bring-up

1. Locate the live Qwen3-8B InfiniLM GPU-worker PID.
2. Run a bounded hcTracer smoke, then one short attach window.
3. Confirm `raw/tracer_out*.json` is produced and normalized.
4. Record whether the trace contains real scheduler markers.

Exit: one normalized profile exists under a run directory with capture
provenance and no invented scheduler ranges.

## M2 - InfiniLM sidecar producer

Add `SidecarSpanRecorder` ranges around:

- schedule/admission
- batch build
- step loop
- model forward
- sampler

Co-emit `tokens_per_iteration`, `batch_size`, and `queue_depth` at the same
window boundary. This is the short-term replacement until InfiniLM emits HCTX
natively.

Exit: baseline and candidate sidecars use the same names, rank ids, and clock
window.

## M3 - Matched A/B integration

Use the existing deterministic promotion wall first. Only invoke hcTracer and
the sidecar when the wall regresses. Feed one baseline and one candidate
profile into `promotion-request`, then write the IO-compatible compare/backlog.

Exit: `runtime.host_graph_command` and scheduler paths remain separate in the
ranked backlog.

## M4 - AutoHelix campaign

AutoHelix consumes `promotion-request-1`, chooses at most one lever, rebuilds
or restarts InfiniLM, then re-observes without profiler overhead.

Exit: iterations are recorded with accepted/rejected levers and unprofiled wall
ratios.

## M5 - Qwen3-8B InfiniLM promotion gate

Promotion is reportable only when the unprofiled primary wall ratio is greater
than the operator threshold, and all wall guardrails pass. A profiled scheduler
span improvement alone is a diagnosis, not a promotion.
