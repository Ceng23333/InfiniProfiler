# metax-9-sg runbook

The host name below is the first development target. Treat all paths and
container names as operator-supplied inputs; InfiniProfiler does not discover
cluster storage by convention.

## Prerequisites

```bash
export HCTRACER_BIN=/path/to/hcTracer
export INFINIPROFILER_ROOT=/path/to/InfiniProfiler
export OUT_DIR=/path/to/suite/profile/$(date +%Y%m%d%H%M%S)
export MODEL=Qwen3-8B
```

Use Python 3.9 or newer. The runtime has no third-party Python dependencies;
pytest is needed only for development.

## Bounded smoke capture

```bash
MODE=smoke DURATION_S=10 \
  "${INFINIPROFILER_ROOT}/scripts/capture_hctracer.sh"
```

## Live attach capture

Resolve the GPU worker PID first. For tensor parallelism, run one rank at a
time or provide rank-specific output directories. Do not attach to every rank
for an unbounded window.

```bash
MODE=attach_serve \
TARGET_PID=<infinilm_gpu_worker_pid> \
CLIENT_WINDOW_CMD='/path/to/bounded_client_window.sh' \
  "${INFINIPROFILER_ROOT}/scripts/capture_hctracer.sh"
```

## Transactional campaign capture

For promotion evidence, use the campaign runner instead of attaching ranks
independently. It starts every required rank tracer before one shared client
window, writes `window_manifest.json`, and stops every tracer even when an
attach or client command fails.

Copy
[`examples/qwen3-8b-metax9-capture-spec.json`](../examples/qwen3-8b-metax9-capture-spec.json),
replace the four PIDs and the client/window paths, then run:

```bash
cd "${INFINIPROFILER_ROOT}"
python3 -m infiniprofiler capture-campaign \
  --spec /path/to/candidate-capture-spec.json \
  --out-dir /path/to/profile/candidate/window0 \
  --hctracer-bin "${HCTRACER_BIN}"
```

After baseline and candidate runs, compare manifests before merging or ranking:

```bash
python3 -m infiniprofiler compare-capture-manifests \
  --baseline /path/to/profile/baseline/window0/window_manifest.json \
  --candidate /path/to/profile/candidate/window0/window_manifest.json \
  --output /path/to/profile/capture_comparison.json

python3 -m infiniprofiler merge-capture \
  --capture-dir /path/to/profile/candidate/window0
```

The merge automatically writes `rank_views.json`. To regenerate it after a
parser update:

```bash
python3 -m infiniprofiler rank-capture \
  --capture-dir /path/to/profile/candidate/window0
```

`MATCHED_GRAPH_REPLAY` means the same graph-replay boundary sequence was found
on every required rank and straggler wait is measured at that granularity.
`COARSE_ONLY` means TP imbalance is a capture-window proxy and straggler wait is
suppressed.

For one raw rank trace, emit exclusive timing attribution with:

```bash
python3 -m infiniprofiler attribute-trace \
  --input /path/to/rank=00/raw/tracer_out-0.json \
  --output /path/to/rank=00/timing_attribution.json
```

Inspect `evidence_mode` before treating launch latency as exact. If hcTracer did
not export correlation IDs, output has `correlation_status=UNKNOWN` and adjacent
timestamp pairing is only a guess.

Before selecting a graph lever, build the source map from the exact candidate
source pin:

```bash
python3 -m infiniprofiler build-source-map \
  --source-root /path/to/infinilm-candidate \
  --source-pin "${CANDIDATE_PIN}" \
  --rules "${INFINIPROFILER_ROOT}/examples/source-map-rules.json" \
  --required-symbol 'MhaKVCache::supports_device_graph_capture' \
  --output /path/to/profile/source-map.json
```

Then pass `--timing-attribution`, `--source-map`, and `--rank-views` to
`promotion-request-v2`. Without those graph inputs, InfiniProfiler will not turn
an aggregate graph ratio into a graph lever.

Create the lever contract with the selected path, expected signals, source or
deployment files, correctness gate, rollback, and unprofiled wall threshold.
Then emit the request:

```bash
python3 -m infiniprofiler promotion-request-v2 \
  --profile /path/to/profile/candidate.json \
  --wall-compare /path/to/suite/compare.json \
  --lever-contract /path/to/lever.json \
  --window-id "${WINDOW_ID}" \
  --profile-ref /path/to/profile/candidate.json \
  --before-profile /path/to/profile/before.json \
  --before-profile-ref /path/to/profile/before.json \
  --campaign-id "${CAMPAIGN_ID}" \
  --model "${MODEL}" --host "${HOST}" --device-track mars \
  --baseline-pin "${BASELINE_PIN}" --candidate-pin "${CANDIDATE_PIN}" \
  --iteration "${ITERATION}" \
  --timing-attribution /path/to/timing.json \
  --source-map /path/to/source-map.json \
  --rank-views /path/to/rank_views.json \
  --output /path/to/autohelix-request-v2.json
```

After the candidate rebuild and matched before/after profiles:

```bash
python3 -m infiniprofiler evaluate-mechanism \
  --request /path/to/autohelix-request-v2.json \
  --before-profile /path/to/profile/before.json \
  --after-profile /path/to/profile/after.json \
  --before-profile-ref /path/to/profile/before.json \
  --after-profile-ref /path/to/profile/after.json \
  --output /path/to/autohelix-request-v2-evaluated.json
```

AutoHelix may use the mechanism classification to accept, reject, or switch a
lever, but only a later unprofiled wall ratio above the threshold can promote.

The current runner requires one unique live PID per rank. If InfiniLM implements
TP ranks as threads in one process, do not attach the same PID multiple times and
do not fabricate rank PIDs. Treat that as one process capture until rank-aware
thread splitting is implemented.

## Naive sidecar capture

When InfiniLM has not yet emitted HCTX ranges, use `SidecarSpanRecorder` inside
the serving process with one `window_id` shared by the baseline and candidate
window. Emit only exact canonical names: `infinilm:queue_wait`,
`infinilm:admission`, `infinilm:batch_build`, `infinilm:schedule`,
`infinilm:forward_step`, and `infinilm:sample`.

Pass structured fields such as `iteration`, `queue_depth`, `batch_size`,
`tokens_per_iteration`, `acceptance_count`, and `preempt_count` directly on the
span. Use `NullSpanRecorder` when profiling is disabled so the engine has no
sidecar serialization or filesystem cost.

The producer writes explicit ranges; it does not sample call stacks. The process
sampler is Linux-only and is a coarse CPU proxy.

## Emit the AutoHelix request

```bash
cd "${INFINIPROFILER_ROOT}"
SOURCE=hctracer \
INPUT="${OUT_DIR}/raw/tracer_out-0.json" \
WALL_COMPARE=/path/to/suite/compare.json \
CAMPAIGN_ID=qwen3-8b-metax9-sg \
BASELINE_PIN=<frozen-vllm-pin> \
CANDIDATE_PIN=<infinilm-worktree-or-tag> \
ITERATION=0 \
  ./scripts/run_autohelix_measure.sh
```

The command stops after writing `profile/autohelix_request.json`. AutoHelix
must select one lever, rebuild/restart the candidate, and run a fresh unprofiled
wall comparison before any promotion claim.
