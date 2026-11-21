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
| Source mapper | Engine-neutral configured source discovery and exact line provenance |
| Diagnostics | Compare baseline/candidate families and rank evidence |
| Promotion bridge | Emit deterministic facts and a one-lever request for AutoHelix |
| Mechanism bridge | Evaluate declared before/after signals without granting promotion |

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

The TP layer aggregates each rank independently. It matches graph-replay
boundaries by sequence when all ranks expose the same boundary count. At that
granularity it reports rank busy time, start skew, last-finish skew, straggler
wait, and imbalance ratio. If boundaries cannot be matched, TP imbalance remains
coarse and straggler wait is suppressed rather than fabricated.

Source mapping is rule-driven and engine-neutral. InfiniProfiler does not assume
an engine family in its core; campaigns supply exact symbols and provider rules.
A graph lever is selected only with paired graph replay timing, a source
guard/provider hit, or rank-aware graph-replay evidence. A large aggregate graph
ratio alone is not actionable.

The v2 mechanism bridge evaluates only signals declared by the lever contract.
Before/after artifacts must share window, campaign, model, source-root identity,
and rank count. Timing movement is inconclusive unless both sides are explicitly
correlated. Classification is `MOVED`, `NO_CHANGE`, `REGRESSED`, or
`INCONCLUSIVE`; profiled mechanism movement never satisfies the wall KPI.

## Naive fallbacks

The sidecar parser accepts a small JSONL contract:

```json
{"type":"span","span_id":1,"name":"infinilm:schedule","window_id":"window-0","start_ns":1,"end_ns":2,"tid":42,"rank":0}
{"type":"metric","name":"tokens_per_iteration","value":31.5,"rank":0}
```

Only exact canonical names become scheduler facts. Unknown scheduler-looking
names remain warnings. A sidecar stream without one accepted window identity
cannot enter a promotion request.

The process sampler is intentionally simple and only uses `/proc`. It samples
per-thread CPU time, voluntary/involuntary context switches, and command lines.
It does not claim to reconstruct call stacks or model-level batch intent, but it
gives the promotion loop a bounded CPU proxy when HCTX is unavailable.

## AutoHelix gate

The promotion bridge consumes wall metrics and normalized profiles, computes a
diagnosis, and emits `promotion-request-1`. AutoHelix owns lever application,
rebuild, acceptance, rejection, and budget. The gate closes only when an
unprofiled wall KPI reaches the operator-defined promotion threshold.
