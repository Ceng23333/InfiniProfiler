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
staging uses Python's built-in sqlite3 module.
