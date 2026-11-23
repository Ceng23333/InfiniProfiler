"""Rank observed evidence without inventing wall or kernel ratios."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..models import ProfileDocument, TimingAttributionFields


HOST_FIRST_WALL = 0.90
HOST_COUNT_MIN = 2.0
HOST_COMMAND_MIN = 2.0
HOST_GAP_MIN = 1.5
TIMING_RATIO_MIN = 1.15
PACKING_RATIO_MAX = 0.85
QUEUE_RATIO_MIN = 1.15
# Matched CU-bucket EngineMetric ratio (candidate/baseline) for kernel.material.
# Only fires when wall is behind HOST_FIRST_WALL; ratios are never invented.
KERNEL_MATERIAL_RATIO_MIN = 1.15
CU_BUCKET_METRIC_PREFIX = "cu_bucket."


LEVERS = {
    "runtime.sync_wait": "inspect host/device synchronization and blocked waits",
    "runtime.api_submit": "inspect CUDA/MACA runtime API submit volume and batching",
    "runtime.host_gap": "inspect host idle gaps around submits; not packing effort",
    "runtime.host_graph_command": "inspect graph policy, host sync, and launch batching",
    "engine.batch_packing": "tune admission/batch packing from engine metrics alone",
    "engine.queue_depth": "inspect admission queue depth from engine metrics alone",
    "graph.source_guard": "disable or narrow the source-level graph-capture guard",
    "graph.provider_selection": "inspect provider selection and graph-compatible fallback path",
    "graph.replay_exclusive": "inspect graph replay device work in the selected source scope",
    "graph.tp_tail": "inspect graph-replay tail latency across TP ranks",
    "kernel.material": "inspect the highest-ratio material device kernel bucket",
}


def _number(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _value(container: Any, name: str) -> Any:
    if isinstance(container, Mapping):
        return container.get(name)
    return getattr(container, name, None)


def _as_timing(value: Any) -> Optional[TimingAttributionFields]:
    if value is None:
        return None
    if isinstance(value, TimingAttributionFields):
        return value
    if isinstance(value, Mapping):
        return TimingAttributionFields.from_mapping(dict(value))
    return None


def primary_wall_ratio(compare: Dict[str, Any]) -> Optional[float]:
    """Read common IO/AIPerf wall payloads without changing their direction."""
    walls = compare.get("walls")
    if isinstance(walls, list):
        primary = next(
            (item for item in walls if isinstance(item, dict) and item.get("primary")),
            None,
        )
        if primary is None and walls and isinstance(walls[0], dict):
            primary = walls[0]
        if primary:
            ratio = _number(
                primary.get("ratio") or primary.get("WALL_DECODE_RATIO")
            )
            if ratio is not None:
                return ratio
    for key in ("ratio", "WALL_DECODE_RATIO", "wall_ratio", "tok_s_ratio"):
        ratio = _number(compare.get(key))
        if ratio is not None:
            return ratio
    return None


def host_first_fired(profile: ProfileDocument, wall_ratio: Optional[float]) -> bool:
    if wall_ratio is None or wall_ratio >= HOST_FIRST_WALL:
        return False

    def ratio(family: str) -> float:
        signal = profile.host(family)
        return float(signal.ratio) if signal and signal.ratio is not None else 0.0

    count_hit = max(ratio("driver"), ratio("runtime"), ratio("api")) >= HOST_COUNT_MIN
    command_hit = max(ratio("command"), ratio("api")) >= HOST_COMMAND_MIN
    gap_hit = ratio("gap") >= HOST_GAP_MIN
    return count_hit or command_hit or gap_hit


def _metric_ratio(profile: ProfileDocument, names: Sequence[str]) -> Optional[float]:
    values = [profile.metric(name).ratio for name in names if profile.metric(name)]
    return next((value for value in values if value is not None), None)


def _timing_ratio(candidate: float, baseline: Optional[float]) -> Optional[float]:
    if baseline is None or baseline <= 0:
        return None
    return candidate / baseline


def _resolve_timing(
    profile: ProfileDocument,
    graph_timing: Optional[Any],
) -> Optional[TimingAttributionFields]:
    return _as_timing(graph_timing) or profile.timing()


def _baseline_timing(profile: ProfileDocument) -> Optional[TimingAttributionFields]:
    provenance = profile.capture.provenance or {}
    raw = provenance.get("baseline_timing_attribution")
    return _as_timing(raw)


def rank_paths(
    profile: ProfileDocument,
    *,
    wall_ratio: Optional[float] = None,
    exhausted: Optional[Sequence[str]] = None,
    graph_timing: Optional[Any] = None,
    graph_source: Optional[Any] = None,
    graph_ranks: Optional[Any] = None,
) -> Dict[str, Any]:
    excluded = set(exhausted or [])
    ranked: List[Dict[str, Any]] = []
    seen = set(excluded)

    timing = _resolve_timing(profile, graph_timing)
    baseline_timing = _baseline_timing(profile)

    graph_timing_mode = _value(timing, "evidence_mode") if timing else None
    graph_launch_time = float(_value(timing, "graph_launch_us") or 0.0) if timing else 0.0
    graph_replay_time = (
        float(_value(timing, "graph_replay_device_us") or 0.0) if timing else 0.0
    )
    graph_timing_paired = (
        timing is not None
        and graph_timing_mode == "EXCLUSIVE_UNION_WITH_EXPLICIT_CORRELATION"
        and graph_launch_time > 0.0
        and graph_replay_time > 0.0
    )

    graph_guard_symbols: List[Mapping[str, Any]] = []
    graph_providers: List[Mapping[str, Any]] = []
    if graph_source is not None:
        for symbol in _value(graph_source, "symbols") or []:
            if isinstance(symbol, Mapping) and symbol.get("kind") == "graph_guard":
                graph_guard_symbols.append(symbol)
        graph_providers = list(_value(graph_source, "providers") or [])

    graph_tail = False
    if graph_ranks is not None:
        for rank_view in _value(graph_ranks, "rank_views") or []:
            if (
                _value(rank_view, "boundary_status") == "MATCHED_GRAPH_REPLAY"
                and float(_value(rank_view, "tp_imbalance_ratio") or 0.0) >= 1.2
            ):
                graph_tail = True
                break

    graph_path: Optional[Dict[str, Any]] = None
    if graph_guard_symbols:
        files = sorted(
            {
                str(symbol.get("file"))
                for symbol in graph_guard_symbols
                if symbol.get("file")
            }
        )
        graph_path = {
            "path_id": "graph.source_guard",
            "reasons": [
                "source map contains an explicit graph-capture guard",
                f"source files={','.join(files)}",
            ],
            "lever": LEVERS["graph.source_guard"],
            "source_files": files,
        }
    elif graph_providers:
        providers = sorted(
            str(provider.get("provider"))
            for provider in graph_providers
            if provider.get("provider")
        )
        graph_path = {
            "path_id": "graph.provider_selection",
            "reasons": [
                "source map contains graph-relevant provider selection rules",
                f"providers={','.join(providers)}",
            ],
            "lever": LEVERS["graph.provider_selection"],
        }
    elif graph_timing_paired:
        graph_path = {
            "path_id": "graph.replay_exclusive",
            "reasons": [
                "explicitly correlated graph launch and replay device work",
                f"graph_launch_us={graph_launch_time:.3f}",
                f"graph_replay_device_us={graph_replay_time:.3f}",
            ],
            "lever": LEVERS["graph.replay_exclusive"],
        }
    elif graph_tail:
        graph_path = {
            "path_id": "graph.tp_tail",
            "reasons": ["matched graph-replay boundaries show TP tail evidence"],
            "lever": LEVERS["graph.tp_tail"],
        }

    if graph_path is not None:
        graph_path.update(
            {
                "rank": len(ranked) + 1,
                "priority_rule": 1,
                "score": float(graph_replay_time or graph_launch_time or 0.0),
            }
        )
        ranked.append(graph_path)
        seen.add(str(graph_path["path_id"]))

    # Priority 2: sync wait from matched timing ratios.
    if timing is not None and "runtime.sync_wait" not in seen:
        sync_ratio = _timing_ratio(
            float(timing.sync_wait_us),
            float(baseline_timing.sync_wait_us) if baseline_timing else None,
        )
        if (
            sync_ratio is not None
            and sync_ratio >= TIMING_RATIO_MIN
            and wall_ratio is not None
            and wall_ratio < HOST_FIRST_WALL
        ):
            seen.add("runtime.sync_wait")
            ranked.append(
                {
                    "rank": len(ranked) + 1,
                    "path_id": "runtime.sync_wait",
                    "priority_rule": 2,
                    "score": sync_ratio,
                    "reasons": [
                        f"matched sync_wait_us ratio={sync_ratio:.3f}",
                        "wall below host-first threshold",
                    ],
                    "lever": LEVERS["runtime.sync_wait"],
                }
            )

    # Priority 3: API submit vs host API ratio.
    if timing is not None and "runtime.api_submit" not in seen:
        api_ratio = _timing_ratio(
            float(timing.api_submit_us),
            float(baseline_timing.api_submit_us) if baseline_timing else None,
        )
        host_api = profile.host("api")
        host_api_ratio = float(host_api.ratio) if host_api and host_api.ratio else None
        submit_hit = api_ratio is not None and api_ratio >= TIMING_RATIO_MIN
        host_hit = host_api_ratio is not None and host_api_ratio >= HOST_COMMAND_MIN
        if submit_hit or host_hit:
            seen.add("runtime.api_submit")
            ranked.append(
                {
                    "rank": len(ranked) + 1,
                    "path_id": "runtime.api_submit",
                    "priority_rule": 3,
                    "score": float(api_ratio or host_api_ratio or 0.0),
                    "reasons": [
                        f"api_submit_us ratio={api_ratio:.3f}"
                        if api_ratio is not None
                        else f"host api ratio={host_api_ratio:.3f}",
                    ],
                    "lever": LEVERS["runtime.api_submit"],
                }
            )

    # Priority 4: host gaps from timing buckets or host gap ratio (not packing).
    if "runtime.host_gap" not in seen:
        before_ratio = after_ratio = None
        if timing is not None:
            gap_before = float(timing.cpu_gap_before_us)
            gap_after = float(timing.cpu_gap_after_us)
            base_before = (
                float(baseline_timing.cpu_gap_before_us) if baseline_timing else None
            )
            base_after = (
                float(baseline_timing.cpu_gap_after_us) if baseline_timing else None
            )
            before_ratio = _timing_ratio(gap_before, base_before)
            after_ratio = _timing_ratio(gap_after, base_after)
        host_gap = profile.host("gap")
        host_gap_ratio = float(host_gap.ratio) if host_gap and host_gap.ratio else None
        elevated = any(
            ratio is not None and ratio >= HOST_GAP_MIN
            for ratio in (before_ratio, after_ratio, host_gap_ratio)
        )
        if elevated and wall_ratio is not None and wall_ratio < HOST_FIRST_WALL:
            seen.add("runtime.host_gap")
            ranked.append(
                {
                    "rank": len(ranked) + 1,
                    "path_id": "runtime.host_gap",
                    "priority_rule": 4,
                    "score": float(before_ratio or after_ratio or host_gap_ratio or 0.0),
                    "reasons": [
                        "elevated cpu_gap_* / host gap with wall regression",
                        "this is host idle, not scheduler packing",
                    ],
                    "lever": LEVERS["runtime.host_gap"],
                }
            )

    # Priority 5: metric-only packing / queue hints (never gated on infinilm: spans).
    tok_ratio = _metric_ratio(
        profile,
        ("tokens_per_iteration", "avg_decoded_tok_per_iter", "tok_per_iter"),
    )
    if (
        tok_ratio is not None
        and tok_ratio < PACKING_RATIO_MAX
        and "engine.batch_packing" not in seen
    ):
        seen.add("engine.batch_packing")
        ranked.append(
            {
                "rank": len(ranked) + 1,
                "path_id": "engine.batch_packing",
                "priority_rule": 5,
                "score": tok_ratio,
                "reasons": [
                    f"tokens/iteration candidate/baseline={tok_ratio:.3f}",
                    "metric-only packing hint",
                ],
                "lever": LEVERS["engine.batch_packing"],
            }
        )

    queue_ratio = _metric_ratio(profile, ("queue_depth", "waiting_requests"))
    if (
        queue_ratio is not None
        and queue_ratio >= QUEUE_RATIO_MIN
        and "engine.queue_depth" not in seen
    ):
        seen.add("engine.queue_depth")
        ranked.append(
            {
                "rank": len(ranked) + 1,
                "path_id": "engine.queue_depth",
                "priority_rule": 5,
                "score": queue_ratio,
                "reasons": [
                    f"queue depth candidate/baseline={queue_ratio:.3f}",
                    "metric-only queue hint",
                ],
                "lever": LEVERS["engine.queue_depth"],
            }
        )

    # Priority 4: material CU-bucket metric with matched baseline ratio.
    if (
        "kernel.material" not in seen
        and wall_ratio is not None
        and wall_ratio < HOST_FIRST_WALL
    ):
        best_name = None
        best_ratio = 0.0
        for metric in profile.metrics:
            name = str(metric.name or "")
            if not name.startswith(CU_BUCKET_METRIC_PREFIX):
                continue
            bucket = name[len(CU_BUCKET_METRIC_PREFIX) :]
            if bucket in ("other", "bf16_copy"):
                continue
            if metric.ratio is None:
                continue
            ratio = float(metric.ratio)
            if ratio > best_ratio:
                best_ratio = ratio
                best_name = name
        if best_name is not None and best_ratio >= KERNEL_MATERIAL_RATIO_MIN:
            seen.add("kernel.material")
            ranked.append(
                {
                    "rank": len(ranked) + 1,
                    "path_id": "kernel.material",
                    "priority_rule": 4,
                    "score": best_ratio,
                    "reasons": [
                        f"highest matched CU-bucket metric {best_name} "
                        f"ratio={best_ratio:.3f} >= {KERNEL_MATERIAL_RATIO_MIN}",
                        "wall below host-first threshold",
                    ],
                    "lever": LEVERS["kernel.material"],
                    "metric": best_name,
                }
            )

    # Fallback host-first path for broad runtime/command evidence.
    if host_first_fired(profile, wall_ratio) and "runtime.host_graph_command" not in seen:
        if not any(
            item["path_id"] in ("runtime.sync_wait", "runtime.api_submit", "runtime.host_gap")
            for item in ranked
        ):
            seen.add("runtime.host_graph_command")
            ranked.append(
                {
                    "rank": len(ranked) + 1,
                    "path_id": "runtime.host_graph_command",
                    "priority_rule": 6,
                    "reasons": [
                        "host runtime/command/gap evidence crossed the documented threshold",
                        "this is not scheduler packing effort",
                    ],
                    "lever": LEVERS["runtime.host_graph_command"],
                }
            )

    if not ranked:
        ranked.append(
            {
                "rank": 1,
                "path_id": "diagnosis.insufficient_evidence",
                "priority_rule": 99,
                "reasons": [
                    "missing matched timing attribution or baseline ratios",
                    "collect a longer matched window with exclusive timing evidence",
                ],
                "lever": "collect missing timing signals before selecting a code lever",
            }
        )

    for rank, item in enumerate(ranked, 1):
        item["rank"] = rank
    return {
        "top_path": ranked[0]["path_id"],
        "host_first_fired": host_first_fired(profile, wall_ratio),
        "timing_evidence": {
            "present": timing is not None,
            "evidence_mode": graph_timing_mode,
            "paired_graph_timing": graph_timing_paired,
        },
        "graph_evidence": {
            "selected": graph_path is not None,
            "paired_graph_timing": graph_timing_paired,
            "source_guard_hits": len(graph_guard_symbols),
            "source_provider_hits": len(graph_providers),
            "tp_tail": graph_tail,
        },
        "ranked": ranked,
        "exhausted": sorted(excluded),
    }
