# InfiniProfiler

InfiniProfiler is a thin profiling abstraction for InfiniLM performance
promotion. It turns vendor traces and lightweight sidecar observations into one
stable profile document, then emits a promotion backlog that AutoHelix can
consume.

The initial target is Mars: live `hcTracer --attach` captures are useful
GPU/runtime timelines, but they do not expose scheduler-level LLM effort unless
the engine emits HCTX ranges or a sidecar supplies them. InfiniProfiler keeps
host signals separate from scheduler signals and never treats launch gaps or
command time as scheduling effort.

See [docs/metax-9-sg-runbook.md](docs/metax-9-sg-runbook.md) for the first live
development target and [docs/roadmap.md](docs/roadmap.md) for the staged
Qwen3-8B promotion gate.

For the promotion-grade feature set, see
[docs/p0-p1-plan.md](docs/p0-p1-plan.md). It defines the smallest Nsight-like
evidence chain needed to make and verify one safe lever choice, not full Nsight
Systems parity.

## Contracts

- `infiniprofiler-profile-1` is the normalized capture document.
- `promotion-request-1` is the deterministic hand-off to AutoHelix.
- Existing `promo-facts-1` remains the durable campaign format. The IO
  compatibility writer only adds facts through documented optional fields.

## Design

InfiniProfiler is organized as a thin universal evidence layer with narrow
engine and vendor adapters. Capture adapters may emit vendor-specific events,
but their output is normalized into stable profile, timeline, rank, source-map,
and promotion contracts. Core ranking and mechanism checks do not assume an
engine family; engine-specific span names, source symbols, providers, pins, and
backend identities are supplied as campaign configuration.

Evidence is separated by mechanism and strength:

| Layer | Evidence |
|---|---|
| Client wall | Unprofiled throughput, TTFT, ITL, success, and promotion KPI |
| Runtime | API submit, launch command, graph launch, synchronization, and CPU gaps |
| Device | Exclusive GPU execution and graph-replay device time |
| Scheduler | Explicit HCTX/marker or sidecar spans; never inferred from host gaps |
| Tensor parallel | Per-rank timing, start skew, tail skew, straggler wait, and imbalance |
| Source | Exact file/line provenance for graph guards, provider selection, and entry points |

The capture path is transactional and matched: every required rank starts before
one shared client window and stops even when another rank, client, or interrupt
fails. Baseline and candidate manifests must agree on campaign, workload, pins,
rank set, warmup policy, and window duration before merging.

Timing facts distinguish aggregate overlap from exclusive interval union and
explicitly paired latency. Streaming hcTracer parsing stages intervals in SQLite,
recovers complete events from truncated captures, and marks adjacent pairing as a
guess unless hcTracer exported a correlation ID.

Promotion is a closed loop with an ownership boundary. InfiniProfiler observes,
normalizes, ranks, declares a lever contract, and evaluates whether the expected
mechanism moved. AutoHelix owns lever selection, rebuild, acceptance, rejection,
and budget. A profiled mechanism result can diagnose progress, but only an
unprofiled wall KPI above the configured threshold can promote.

The v2 request requires a lever contract with expected signals, source or
deployment scope, correctness gate, rollback, and unprofiled wall threshold.
Before/after mechanism evaluation checks matching window, campaign, model,
source-root identity, and rank count. Missing, mismatched, or weakly correlated
evidence becomes `INCONCLUSIVE` rather than an actionable lever.

## Commands

```bash
python -m infiniprofiler normalize \
  --source hctracer --input trace.json --output profile.json

python -m infiniprofiler normalize \
  --source naive-sidecar --input scheduler.jsonl --output profile.json

python -m infiniprofiler promotion-request \
  --profile profile.json --wall-compare compare.json \
  --output autohelix-request.json

python -m infiniprofiler io-compat \
  --profile profile.json --out-dir profile/iteration-0

python -m infiniprofiler validate --input profile.json
python -m infiniprofiler validate --kind promotion-request --input request.json

python -m infiniprofiler capture-campaign \
  --spec capture-spec.json --out-dir profile/window0

python -m infiniprofiler compare-capture-manifests \
  --baseline profile/baseline/window0/window_manifest.json \
  --candidate profile/candidate/window0/window_manifest.json \
  --output profile/capture_comparison.json

python -m infiniprofiler merge-capture --capture-dir profile/candidate/window0
python -m infiniprofiler rank-capture --capture-dir profile/candidate/window0

python -m infiniprofiler build-source-map \
  --source-root /path/to/engine \
  --source-pin engine-candidate \
  --rules examples/source-map-rules.json \
  --required-symbol 'Provider::supports_graph_capture' \
  --output source-map.json

python -m infiniprofiler promotion-request-v2 \
  --profile profile.json --wall-compare compare.json \
  --lever-contract lever.json --window-id window-0 \
  --profile-ref profile/candidate.json \
  --before-profile before.json --before-profile-ref profile/before.json \
  --campaign-id campaign --baseline-pin frozen --candidate-pin candidate \
  --iteration 0 --output request-v2.json

python -m infiniprofiler evaluate-mechanism \
  --request request-v2.json --before-profile before.json \
  --after-profile after.json --before-profile-ref profile/before.json \
  --after-profile-ref profile/after.json --output request-v2-moved.json

python -m infiniprofiler attribute-trace \
  --input trace.json --output timing.json
```

`promotion-request` stops after producing facts and the selected lever. It
never rebuilds InfiniLM or applies an environment lever itself.

## Development

```bash
python -m pip install -e '.[dev]'
pytest
```

The Python implementation has no third-party runtime dependencies; timeline
staging uses Python's built-in sqlite3 module. Engine-specific source symbols
are campaign configuration, not core assumptions.

`promotion-request-v2` is also backend-neutral: baseline and candidate backend
names are campaign inputs rather than fixed engine assumptions.
