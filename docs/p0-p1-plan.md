# P0/P1 feature plan

## Scope

Do not target full Nsight Systems parity. The goal is the smallest evidence
chain that can choose one safe lever per iteration and later prove whether that
lever moved the expected mechanism.

Primary promotion evidence is **exclusive timeline attribution** (the same
families Nsight truth emits). Canonical `infinilm:*` `scheduler_signals` are
not recorded or ranked. Engine metrics may inform packing/queue hints only.
Optional engine NVTX/HCTX ranges are provenance (L3), not a restored span
contract.

## Evidence model

| Layer | Evidence | Source | Rank paths |
|---|---|---|---|
| L1 | `timing_attribution` + host_signals | CUPTI / Chrome / nsys | `runtime.*`, `graph.*` |
| L2 | tokens/iter, queue_depth | sidecar / client metrics | `engine.batch_packing`, `engine.queue_depth` |
| L3 | optional named engine ranges | NVTX / HCTX / sidecar spans | provenance; future `engine.span_*` only |

Never map `cpu_gap_*` / sync / API to packing or scheduler effort.

## Contract strategy

| Artifact | Decision |
|---|---|
| `infiniprofiler-profile-1` | Retired as a document schema; file retains shared `$defs` only. |
| `infiniprofiler-normalize-1` | Timing-first normalize output (`capture`, `host_signals`, `timing_attribution`, `metrics`, `warnings`). |
| `infiniprofiler-profile-2` | Promotion-grade format with matched window, workload identity, rank views, exclusive timing, source map. No `scheduler_signals`. |
| `promotion-request-1` | Frozen AutoHelix smoke hand-off. |
| `promotion-request-2` | Lever contract + optional post-rebuild mechanism check. |
| `naive-sidecar-jsonl-1` | Metrics are L2 evidence; span rows are L3 provenance/warnings only. |

A v2 validator must reject a promotion request that lacks window alignment or a
lever contract.

## P0 acceptance gate

A campaign may enter AutoHelix iteration only when all are true:

1. Baseline and candidate capture manifests match campaign, model, backend pins,
   TP, workload identity, warmup policy, and the referenced wall workload.
2. Every expected rank has a readable capture or the result is explicitly
   `PARTIAL` and rank-sensitive paths are disabled.
3. Timing outputs distinguish API submit, GPU execution, launch latency, sync
   wait, graph launch/replay, and CPU gap. Overlapping sums are not called
   exclusive time.
4. Ranking uses matched baseline/candidate timing ratios (and optional engine
   metrics). Host gaps never become packing evidence.
5. The selected graph-related lever identifies source or deployment files and
   the expected signals to move.
6. Promotion remains based on an unprofiled wall KPI above `1.0`; profiled
   evidence only explains and ranks mechanisms.

## Required timing evidence

| Field | Role (nsys analogue) |
|---|---|
| `api_submit_us` | CUDA/MACA runtime API on CPU |
| `gpu_execution_us` | Device busy |
| `sync_wait_us` | Sync / wait |
| `graph_launch_us` / `graph_replay_device_us` | Graph launch vs replay |
| `cpu_gap_before_us` / `cpu_gap_after_us` | Host idle around submits |
| `launch_latency_us` | Submit→device start |

Matched A/B uses baseline+candidate ratios on these buckets.

## Ranking priority (one path per iteration)

1. Graph paths when paired `graph_launch_us` + `graph_replay_device_us` (or source-map guard/provider).
2. `runtime.sync_wait` when matched `sync_wait_us` ratio ≥ threshold and wall &lt; 0.9.
3. `runtime.api_submit` when matched `api_submit_us` / host api ratio crosses threshold.
4. `runtime.host_gap` for elevated `cpu_gap_*` / host gap with wall regress.
5. Metric-only hints: `engine.batch_packing` / `engine.queue_depth`.
6. Else `diagnosis.insufficient_evidence` citing missing matched timing/baseline.

## Sidecar / optional ranges

Emit engine metrics at the matched window (L2). Free-form span/marker names may
appear in provenance/warnings (L3). They are not a `scheduler_signals` contract.
A real L3 rank path later requires matched A/B ranges, shared window identity,
and explicit ratio thresholds — still without inventing effort from CUDA gaps.

## Mechanism check

The bridge resolves expected signals from timing attribution, rank views, or
engine metrics. Timing movement is inconclusive unless both profiles use
explicitly correlated exclusive attribution.

## NVIDIA A100 acceptance

Host `ssh A100`, vLLM TP4 Qwen3-8B BF16, `compare_nsys_gate.py` PASS on timing
families only. Closed-loop profile-2 + AutoHelix request is built from the same
matched nsys runs. Showroom mcTracer normalize is optional smoke, not the gate.

## Highest-value feature slice

1. matched multi-rank capture,
2. streaming exclusive timing attribution,
3. rank tail/imbalance,
4. source map for graph guard/provider,
5. timing-based lever + mechanism check,
6. optional L3 engine-range provenance without restoring span contracts.

If required timing evidence is missing, InfiniProfiler must return
`insufficient_evidence` rather than inventing scheduler spans from CUDA gaps.
