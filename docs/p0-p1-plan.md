# P0/P1 feature plan

## Scope

Do not target full Nsight Systems parity. The goal is the smallest evidence
chain that can choose one safe lever per iteration and later prove whether that
lever moved the expected mechanism.

The current `profile-1` implementation is a useful parser boundary, but it is
not promotion-grade. It has no normalized matched-window identity, no rank
merge, no interval algebra, no source mapping, and no mechanism check. Those are
new semantics, so they should use new versions rather than silently changing
v1.

## Contract strategy

| Artifact | Decision |
|---|---|
| `infiniprofiler-profile-1` | Freeze. Existing parser and AutoHelix smoke flow remain valid. |
| `infiniprofiler-profile-2` | P0 format with matched window, workload identity, rank views, exclusive timing, scheduler spans, source map, and evidence provenance. |
| `promotion-request-1` | Freeze. |
| `promotion-request-2` | P0 format with a required lever contract and optional post-rebuild mechanism check. |
| `naive-sidecar-jsonl-1` | Extend additively with optional fields; do not rename existing fields. |

Unknown fields remain ignored by v1 consumers. A v2 validator must reject a
promotion request that lacks window alignment or a lever contract.

## P0 acceptance gate

A campaign may enter AutoHelix iteration only when all are true:

1. Baseline and candidate capture manifests match campaign, model, backend pins,
   TP, workload identity, warmup policy, and the referenced wall workload.
2. Every expected rank has a readable capture or the result is explicitly
   `PARTIAL` and rank-sensitive paths are disabled.
3. Timing outputs distinguish API submit, GPU execution, launch latency, sync
   wait, graph launch/replay, and CPU gap. Overlapping sums are not called
   exclusive time.
4. Scheduler paths require explicit HCTX/marker or sidecar spans. Host gaps never
   become scheduler evidence.
5. The selected graph-related lever identifies source or deployment files and
   the expected signals to move.
6. Promotion remains based on an unprofiled wall KPI above `1.0`; profiled
   evidence only explains and ranks mechanisms.

# P0 workstreams

## 1. Matched-window capture

### Data model

Add to profile v2:

```json
{
  "window": {
    "window_id": "required",
    "wall_compare_ref": "path",
    "wall_workload_id": "required",
    "client_started_epoch_ns": 0,
    "client_ended_epoch_ns": 0,
    "warmup_excluded": true,
    "start_skew_ms": 0,
    "duration_ratio": 1.0,
    "alignment": "MATCHED"
  },
  "workload": {
    "campaign_id": "required",
    "model": "Qwen3-8B",
    "backend_baseline_pin": "required",
    "backend_candidate_pin": "required",
    "tensor_parallel": 4,
    "concurrency": 0,
    "prompt_lengths": [],
    "output_lengths": [],
    "batch_size": null,
    "clock_policy": "operator_observed",
    "profiles_source_hash": "required"
  }
}
```

`clock_policy` records observed clocks or the operator policy; it must not
invent a controlled-clock claim.

### Capture orchestration

Implement a campaign runner around `capture_hctracer.sh`:

1. Resolve rank pid/tid mapping before attach.
2. Attach each selected rank and wait until all tracers are ready.
3. Write `window_manifest.json` immediately before the client window.
4. Run one bounded client command for all ranks.
5. Stop every hcTracer with Ctrl+T, not SIGINT.
6. Write the end manifest and collect artifacts into
   `rank=<nn>/raw/tracer_out*.json`.

The wrapper must be transactional: if any required rank fails to start, stop the
started tracers, mark the run `FAILED_RANK_ATTACH`, and do not emit a ranked
promotion request.

### Alignment rule

An unprofiled wall run and a profiled diagnosis run cannot occupy the same
physical time interval. The match is therefore at workload and manifest level:

- `compare.json` contains `workload_id` and the wall request definition.
- Profile `window.wall_workload_id` must equal that id.
- Client start/end bracket the same bounded prompt/output sequence.
- Start skew tolerance is 50 ms by default.
- Duration ratio tolerance is 0.95-1.05 by default.
- Any mismatch emits `WINDOW_MISMATCH` and suppresses lever selection.

### Tests

- Two-rank fixture produces one merged manifest and separate rank artifacts.
- Baseline/candidate identity mismatch is rejected.
- Duration and start-skew violations produce `WINDOW_MISMATCH`.
- Failed rank attach produces a transactional failure and no request.

## 2. Per-rank and merged TP view

### Input layout

```text
profile/run/
  window_manifest.json
  rank=00/raw/tracer_out*.json
  rank=01/raw/tracer_out*.json
  rank=02/raw/tracer_out*.json
  rank=03/raw/tracer_out*.json
```

### Normalization

Each rank is normalized independently first, preserving:

- campaign id and window id
- rank
- pid
- tid
- stream
- correlation id when hcTracer exports one
- device/kernel name
- start/end epoch ns
- event category

The merged view is derived from rank-local aggregates. Do not concatenate raw
events as the durable product.

### TP metrics

For each synchronized iteration or graph replay boundary, compute:

| Metric | Definition |
|---|---|
| `rank_gpu_busy_us` | Union of kernel/memcpy/graph execution intervals on one rank |
| `rank_last_finish_skew_us` | Rank end minus minimum rank end |
| `rank_start_skew_us` | Rank start minus minimum rank start |
| `straggler_wait_us` | Minimum rank end minus that rank end, clamped to zero |
| `tp_imbalance_ratio` | Maximum rank busy divided by median rank busy |

If synchronized iteration boundaries cannot be identified, emit
`correlation=UNKNOWN`. Then report coarse rank imbalance as a warning, but do
not claim straggler wait.

### Tests

- Rank local totals never inherit another rank's events.
- Merged execution uses interval union and does not double-count overlap.
- Missing rank disables TP-specific paths.
- A synthetic straggler produces nonzero `straggler_wait_us`.

## 3. Exclusive-time attribution

### Principle

Aggregate `sum(dur)` is diagnostic only. Every P0 timing fact must state whether
it is `summed_overlap`, `exclusive_union`, or `paired_latency`.

### Required outputs

| Field | Method |
|---|---|
| `api_submit_us` | Exclusive union of runtime/driver submit intervals |
| `gpu_execution_us` | Exclusive union of non-nested device execution intervals |
| `launch_latency_us` | GPU start minus matching submit end |
| `sync_wait_us` | Exclusive union of synchronization and blocked host wait intervals |
| `graph_launch_us` | External graph launch/replay API interval union |
| `graph_replay_device_us` | Device intervals attributed to graph replay |
| `cpu_gap_before_us` | Preceding host submit end to GPU start when pairing exists |
| `cpu_gap_after_us` | GPU end to next host submit start when pairing exists |
| `unpaired_event_count` | Count excluded from latency attribution |

`graph=31x` may remain a summary ratio, but it is not a lever. The request must
say whether the material interval is graph replay, launch batching, guard
fallback, sync wait, CPU gap, or rank imbalance.

### Interval engine

Implement interval union separately for:

- one pid/thread
- one GPU stream
- one rank
- graph replay subtree

Handle three cases explicitly:

1. overlapped intervals on different streams,
2. nested graph replay and child kernels,
3. host submit and device execution correlation.

If hcTracer does not expose correlation, use timestamp adjacency only for an
explicit `adjacent_guess`, and carry `unpaired_event_count` plus a warning. It
must not be presented as exact launch latency.

### Tests

Use deterministic microsecond fixtures:

- two same-stream overlaps produce a union, not a sum;
- different-stream overlaps produce rank union correctly;
- graph parent and child kernel do not double-count device time;
- unmatched submit/GPU pairs increment `unpaired_event_count`;
- a 31x graph ratio with low exclusive graph time cannot select
  `runtime.host_graph_command` by itself.

## 4. Scheduler spans

### Canonical names

Use these stable names in HCTX and sidecar producers:

```text
infinilm:queue_wait
infinilm:admission
infinilm:batch_build
infinilm:schedule
infinilm:forward_step
infinilm:sample
```

Each span carries optional `iteration`, `queue_depth`, `batch_size`,
`tokens_per_iteration`, `acceptance_count`, and `preempt_count`.

### Sidecar producer

Extend `SidecarSpanRecorder` to accept structured fields and emit a monotonic
`span_id`, `iteration`, and window id. Keep JSONL append-only so the engine can
write continuously without holding profiler state.

### Parser rules

- Exact canonical prefixes become scheduler signals.
- Unknown scheduler-looking names become warnings, not scheduler facts.
- Host gaps cannot map to `scheduler.*`.
- A sidecar file with no matched window cannot enter a P0 request.

### InfiniLM integration

Instrument in this order:

1. scheduler loop around admission/schedule/batch build,
2. executor around forward step,
3. sampler around sample,
4. queue wait at request insertion/admission,
5. per-iteration metrics at the same boundary.

The producer must be safe when profiling is disabled: append mode plus a null
sink must add no JSON serialization cost.

## 5. Source mapping

### Source map artifact

Create `source-map-1.json` from the exact candidate source pin:

```json
{
  "schema_version": "source-map-1",
  "source_pin": "required",
  "source_root_hash": "required",
  "symbols": [
    {
      "symbol": "MhaKVCache::supports_device_graph_capture",
      "file": "path/from/source/root",
      "line_start": 0,
      "kind": "graph_guard"
    }
  ],
  "providers": [
    {
      "provider": "flash_attn|fallback|native",
      "patterns": ["kernel or symbol regex"],
      "files": []
    }
  ]
}
```

### Mapping pass

Add a source mapper that scans the candidate source tree and records:

- attention/GEMM provider selection,
- graph capture eligibility,
- graph guard functions,
- scheduler loop files,
- deployment entry points.

`MhaKVCache::supports_device_graph_capture` is a required campaign-specific
symbol for the current graph path. The source mapper must record its exact file
and line.

### Ranking rule

A graph path can be selected only when at least one is present:

1. paired graph launch/replay exclusive time,
2. an explicit fallback/guard span or log marker,
3. a source-map hit for the guard or provider branch,
4. rank-aware evidence that graph replay itself is material.

The resulting path should prefer `graph.source_guard` or
`graph.provider_selection` over a deployment-only
`runtime.host_graph_command` when source evidence exists.

## 6. Lever contract and mechanism check

Add `promotion-request-2`:

```json
{
  "lever_contract": {
    "lever_id": "required",
    "top_path": "required",
    "expected_signals": [
      {"name": "graph_launch_us", "direction": "decrease", "min_ratio": 0.8}
    ],
    "source_files": [],
    "deployment_files": [],
    "correctness_gate": "required",
    "wall_gate": {"profiled": false, "ratio_gt": 1.0},
    "rollback": "required"
  },
  "mechanism_check": {
    "status": "NOT_RUN",
    "before_profile_ref": "",
    "after_profile_ref": "",
    "results": []
  }
}
```

Mechanism status is one of:

- `MOVED`: every expected signal crosses its threshold in profiled before/after
  matched windows.
- `NO_CHANGE`: wall changed but mechanism signals did not.
- `REGRESSED`: mechanism moves the wrong way.
- `INCONCLUSIVE`: window mismatch, missing rank, or too many unpaired events.

AutoHelix must not claim success from wall movement alone. The durable iteration
record contains both mechanism check and unprofiled wall result.

## P0 sequencing

| Milestone | Deliverable | Gate |
|---|---|---|
| P0-A | Profile/request v2 schemas and validation fixtures (**done**) | Invalid window or missing lever contract rejected by `infiniprofiler validate` |
| P0-B | Matched multi-rank wrapper and manifest merge (**done**) | Two-rank capture aligns and failed rank aborts cleanly; one-PID/thread-rank splitting is explicitly not supported yet |
| P0-C | Streaming parser and interval engine (**done**) | Exclusive union, graph nesting, explicit pairing, adjacent-guess warnings, and truncated streaming fixtures pass |
| P0-D | Rank merge and TP imbalance facts (**done**) | Graph-replay boundary matching, coarse fallback, and synthetic straggler tests pass |
| P0-E | InfiniLM sidecar spans and parser rules (**done**) | Canonical names, window identity, null producer, unknown-span warnings, and metric-only suppression tests pass |
| P0-F | Source map and graph ranking rule (**done**) | Engine-neutral rules, exact source hits, paired graph replay gate, and source-path precedence tests pass |
| P0-G | Mechanism check bridge | AutoHelix iteration records wall and mechanism independently |

P0-C starts immediately after contract freeze because exclusive graph
attribution is the highest-value feature.

# P1 workstreams

## 1. Gap classification

Classify exclusive CPU gaps with evidence precedence:

1. explicit scheduler/queue span,
2. synchronization/wait event,
3. graph guard or fallback marker,
4. TP tail/rank imbalance,
5. launch latency with a submit/device pair,
6. memcpy or allocator stall,
7. CPU busy while GPU waits,
8. `UNKNOWN`.

Do not allow precedence inversion: a scheduler span wins over a coincident
host gap; a graph guard marker wins over generic graph ratio.

Required output:

```json
{
  "gap_classes": [
    {
      "class": "graph_guard_fallback",
      "exclusive_us": 0,
      "count": 0,
      "confidence": "EXPLICIT",
      "evidence": ["span", "source_map", "trace_event"]
    }
  ]
}
```

## 2. Aligned `/proc` sampling

Extend the sampler to:

- start and stop from `window_manifest.json`,
- emit epoch ns and monotonic ns for every row,
- record thread state from `/proc/<pid>/task/<tid>/status`,
- aggregate by 100 ms default bins,
- correlate by pid/tid and time bin,
- emit CPU busy, sleeping, voluntary/involuntary context switches, and runqueue
  hint only when available.

Interpretation:

- busy CPU while GPU waits favors launch/source work;
- sleeping CPU while queue depth rises favors scheduler starvation;
- one thread/rank tails the others favors TP imbalance;
- all ranks sleep while queue grows favors admission policy.

This is a lightweight proxy, not a call-stack sampler.

## 3. Streaming trace parsing

Replace full-document loading for profile v2 while retaining v1 behavior:

1. Memory-map or chunk-read the hcTracer JSON.
2. Incrementally decode complete `traceEvents` objects.
3. Send each event to an aggregator callback.
4. Recover complete events from a truncated tail.
5. Spill sortable event records to SQLite or compact temporary files.
6. Compute interval algebra after time/rank sorting.

Memory target: the resident parser buffer must not grow with trace size. A
synthetic multi-hundred-MB fixture should run with bounded RSS in CI or a
nightly performance job.

The durable output remains compact facts plus optional indexed artifacts, not
the original raw trace.

## 4. Noise and repeatability

Extend wall handling to store repeated observations:

```json
{
  "wall_repeats": [
    {"rep": 0, "ratio": 1.0, "accepted": true}
  ],
  "wall_summary": {
    "rep_count": 3,
    "median_ratio": 1.0,
    "cv": 0.0,
    "promotion_candidate": false
  }
}
```

Rules:

- default three successful unprofiled repeats,
- exclude configured warmup,
- report median and coefficient of variation,
- do not use profiled repeats for promotion,
- require median unprofiled ratio `> 1.0` and all guardrails passing.

# Highest-value feature slice

Implement this first after P0-A:

1. two-rank matched hcTracer capture,
2. streaming event normalization,
3. interval-union graph attribution,
4. rank tail/imbalance split,
5. source map for graph guard/provider selection,
6. one source-scope graph lever plus mechanism-check contract.

The minimum question answered is:

> Is the graph blocker caused by graph replay itself, graph guard fallback,
> CPU launch batching, synchronization wait, or TP straggling, and did the next
> source-scope patch move that exact mechanism?

If any input required to answer that question is missing, InfiniProfiler must
return `insufficient_evidence` rather than recommending another deployment-only
knob.
