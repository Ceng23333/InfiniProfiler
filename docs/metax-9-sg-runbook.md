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

## Naive sidecar capture

When InfiniLM has not yet emitted HCTX ranges, run the engine sidecar producer
inside the serving process. The producer writes explicit ranges; it does not
sample call stacks. The process sampler is Linux-only and is a coarse CPU
proxy.

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
