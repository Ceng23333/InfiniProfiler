# Architecture

## Goals

InfiniProfiler separates capture, normalization, diagnosis, and promotion
control. A vendor adapter may only produce evidence it observed. Host runtime
events and CUDA idle gaps are never relabeled as scheduler or packing effort.

## Evidence layers

| Layer | Evidence | Source | Rank paths |
|---|---|---|---|
| L1 Nsight-truth (required) | `timing_attribution` + `host_signals` | CUPTI / Chrome timeline / nsys SQLite | `runtime.*`, `graph.*` |
| L2 Engine metrics (optional) | tokens/iter, queue_depth, batch_size | sidecar / client metrics | `engine.batch_packing`, `engine.queue_depth` (hints only) |
| L3 Engine ranges (optional) | named schedule/admission/batch/forward ranges | NVTX / HCTX / sidecar span rows | provenance today; future `engine.span_*` only — never restored `scheduler_signals` / `infinilm:*` contracts |

Rules:

1. L1 is required for promotion diagnosis. Missing matched timing fails closed
   to `diagnosis.insufficient_evidence` (or metric-only L2 hints).
2. Never map `cpu_gap_*`, sync wait, or API submit volume to packing / scheduler
   effort.
3. L3 is additive. Free-form ranges land in provenance/warnings (or an optional
   schema field later). Profile-1 `scheduler_signals` smoke stays retired.

## Component layers

| Layer | Responsibility |
|---|---|
| Capture adapters | Start or attach to a bounded capture, collect raw artifacts |
| Normalizer | Repair and classify traces into timing-first normalize + timeline facts |
| Naive tools | Metrics sidecar and `/proc` sampler when stacks are unavailable |
| Source mapper | Engine-neutral configured source discovery and exact line provenance |
| Diagnostics | Compare baseline/candidate families and rank evidence |
| Promotion bridge | Emit deterministic facts and a one-lever request for AutoHelix |
| Mechanism bridge | Evaluate declared before/after signals without granting promotion |

## Profile document

Normalize output (`infiniprofiler-normalize-1`) contains:

- `capture`: source, mode, pid, ranks, window, and truncation provenance.
- `host_signals`: driver/runtime/command/api/gap/graph families. Ratios are
  emitted only when a comparable baseline exists.
- `timing_attribution`: exclusive `api_submit_us`, `gpu_execution_us`,
  `sync_wait_us`, graph launch/replay, and CPU gap buckets.
- `metrics`: engine/client metrics such as tokens per iteration, batch size,
  and queue depth.
- `warnings`: lost data, unresolved pid, optional free-form markers, or
  fallback behavior.

Promotion-grade `infiniprofiler-profile-2` adds matched window, workload
identity, rank views, and source map. It does not carry `scheduler_signals`.

## hcTracer / nsys adapters

The hcTracer/mcTracer adapter reads Chrome-trace JSON, repairs truncated tails,
maps runtime/driver/kernel/memcpy/graph events to host families, and always
runs the streaming timeline builder so normalize emits `timing_attribution`.
Optional free-form marker names may appear in warnings/provenance only.

The nsys adapter maps CUPTI SQLite rank truth into the same timing buckets and
can export per-rank Chrome traces for the shared timeline engine.

## Promotion timeline

The timeline path streams `traceEvents` in bounded chunks and stages sortable
intervals in SQLite. It computes exclusive interval unions for submit, GPU
execution, synchronization, graph launch, and graph replay device work. It
pairs submit/device intervals when a correlation ID is present; otherwise it
marks adjacent timestamp pairing as a guess and reports the unpaired count.

The TP layer aggregates each rank independently. It matches graph-replay
boundaries by sequence when all ranks expose the same boundary count. At that
granularity it reports rank busy time, start skew, last-finish skew, straggler
wait, and imbalance ratio. If boundaries cannot be matched, TP imbalance remains
coarse and straggler wait is suppressed rather than fabricated.

Source mapping is rule-driven and engine-neutral. A graph lever is selected
only with paired graph replay timing, a source guard/provider hit, or
rank-aware graph-replay evidence. A large aggregate graph ratio alone is not
actionable.

The mechanism bridge evaluates only signals declared by the lever contract.
Before/after artifacts must share window, campaign, model, source-root identity,
and rank count. Timing movement is inconclusive unless both sides are explicitly
correlated. Classification is `MOVED`, `NO_CHANGE`, `REGRESSED`, or
`INCONCLUSIVE`; profiled mechanism movement never satisfies the wall KPI.

## Naive fallbacks

The sidecar parser accepts a small JSONL contract:

```json
{"type":"metric","name":"tokens_per_iteration","value":31.5,"rank":0,"window_id":"window-0"}
{"type":"span","span_id":1,"name":"optional-marker","window_id":"window-0","start_ns":1,"end_ns":2,"tid":42,"rank":0}
```

Metric rows are L2 promotion hints. Span rows are L3 provenance/warnings only —
they never become a `scheduler_signals` contract. A sidecar stream without one
accepted window identity cannot enter a promotion request.

The process sampler is intentionally simple and only uses `/proc`. It samples
per-thread CPU time, voluntary/involuntary context switches, and command lines.
It does not claim to reconstruct call stacks or model-level batch intent.

## AutoHelix gate

The promotion bridge consumes wall metrics and normalized profiles, computes a
diagnosis, and emits a promotion request with `stop_before_promote=true`.
AutoHelix owns lever application, rebuild, acceptance, rejection, and budget.
The gate closes only when an unprofiled wall KPI reaches the operator-defined
promotion threshold.
