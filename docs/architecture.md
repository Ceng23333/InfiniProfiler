# Architecture

## Goals

InfiniProfiler separates capture, normalization, diagnosis, and promotion
control. A vendor adapter may only produce evidence it observed. In particular,
host runtime events cannot be relabeled as scheduler effort.

## Layers

| Layer | Responsibility |
|---|---|
| Capture adapters | Start or attach to a bounded capture, collect raw artifacts |
| Normalizer | Repair and classify traces into v1 and promotion-grade timeline facts |
| Naive tools | Provide fallback signals when HCTX/CPU stacks are unavailable |
| Diagnostics | Compare baseline/candidate families and rank evidence |
| Promotion bridge | Emit deterministic facts and a one-lever request for AutoHelix |

## Profile document

`infiniprofiler-profile-1` contains:

- `capture`: source, mode, pid, ranks, window, and truncation provenance.
- `host_signals`: driver/runtime/command/api/gap/graph families. Ratios are
  emitted only when a comparable baseline exists.
- `scheduler_signals`: explicit HCTX/marker spans or sidecar observations.
- `metrics`: engine/client metrics such as tokens per iteration, batch size,
  queue depth, TTFT, ITL, and throughput.
- `warnings`: lost data, unresolved pid, missing annotation producer, or
  fallback behavior.

## hcTracer adapter

The hcTracer adapter reads Chrome-trace JSON and repairs the common truncated
tail. It maps runtime/driver/kernel/memcpy/graph events to host families. It
also extracts `ph=X` marker ranges whose names contain `schedule`, `step`,
`batch`, `forward`, or `sample` into scheduler signals.

If no marker producer is present, the adapter emits a warning and no synthetic
scheduler spans. This is deliberate.

## Promotion timeline

The v2 timeline path streams `traceEvents` in bounded chunks and stages sortable
intervals in SQLite. It computes exclusive interval unions for submit, GPU
execution, synchronization, graph launch, and graph replay device work. It
pairs submit/device intervals when hcTracer exports a correlation ID; otherwise
it marks adjacent timestamp pairing as a guess and reports the unpaired count.

## Naive fallbacks

The sidecar parser accepts a small JSONL contract:

```json
{"type":"span","name":"scheduler.step","start_ns":1,"end_ns":2,"tid":42,"rank":0}
{"type":"metric","name":"tokens_per_iteration","value":31.5,"rank":0}
```

The process sampler is intentionally simple and only uses `/proc`. It samples
per-thread CPU time, voluntary/involuntary context switches, and command lines.
It does not claim to reconstruct call stacks or model-level batch intent, but it
gives the promotion loop a bounded CPU proxy when HCTX is unavailable.

## AutoHelix gate

The promotion bridge consumes wall metrics and normalized profiles, computes a
diagnosis, and emits `promotion-request-1`. AutoHelix owns lever application,
rebuild, acceptance, rejection, and budget. The gate closes only when an
unprofiled wall KPI reaches the operator-defined promotion threshold.
