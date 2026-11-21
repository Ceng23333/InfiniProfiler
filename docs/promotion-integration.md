# Promotion integration

## Qwen3-8B / InfiniLM baseline gate

The first end-to-end gate is:

1. Run unprofiled Qwen3-8B vLLM and InfiniLM wall observations.
2. If the primary wall or any guardrail regresses, run matched profiling.
3. Normalize hcTracer and naive sidecar evidence.
4. Emit ranked paths without mixing scheduler effort into `host_graph_command`.
5. Hand one lever to AutoHelix and stop.
6. AutoHelix rebuilds/restarts InfiniLM and repeats the unprofiled measurement.
7. Report promotion only after the unprofiled wall ratio exceeds the configured
   threshold, by default `1.0`.

## Required identity

The request includes stable campaign identity: host, device track, model, TP
degree, baseline/candidate pins, campaign id, iteration id, and wall artifact
paths. Unknown optional fields are preserved so `promo-facts-1` can consume the
payload later.

## Path rules

Scheduler evidence uses a different path namespace from runtime evidence:

- `scheduler.step_control` for step loop overhead.
- `scheduler.batch_packing` for low tokens per iteration or low batch size.
- `scheduler.queue_admission` for queue growth without equivalent running work.
- `runtime.host_graph_command` remains the existing host-first path.

Scheduler paths require exact canonical `infinilm:` spans. Metric-only sidecar
data can enrich a matched scheduler finding but cannot create one. A sidecar
stream with no matched window cannot enter a P0 promotion request.

Rank order is explicit evidence first, wall severity second. The implementation
does not infer a scheduler path from a large `gap` ratio alone.
