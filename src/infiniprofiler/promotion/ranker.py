"""Rank observed evidence without inventing wall or kernel ratios."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..models import ProfileDocument
from ..scheduler import (
    ADMISSION,
    BATCH_BUILD,
    QUEUE_WAIT,
    canonical_scheduler_family,
)


HOST_FIRST_WALL = 0.90
HOST_COUNT_MIN = 2.0
HOST_COMMAND_MIN = 2.0
HOST_GAP_MIN = 1.5
SCHEDULER_RATIO_MIN = 1.15
PACKING_RATIO_MAX = 0.85


LEVERS = {
    "scheduler.step_control": "reduce engine step-loop overhead; inspect schedule/sample/forward spans",
    "scheduler.batch_packing": "tune admission/batch packing; raise tokens per iteration",
    "scheduler.queue_admission": "inspect admission queue and request dispatch policy",
    "runtime.host_graph_command": "inspect graph policy, host sync, and launch batching",
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


def scheduler_findings(profile: ProfileDocument) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []

    def explicit_span(*names: str) -> bool:
        return any(
            signal.name in names
            and canonical_scheduler_family(signal.name) is not None
            and signal.source in ("naive-sidecar", "hctracer.marker")
            for signal in profile.scheduler_signals
        )

    tok_ratio = _metric_ratio(
        profile,
        ("tokens_per_iteration", "avg_decoded_tok_per_iter", "tok_per_iter"),
    )
    if (
        tok_ratio is not None
        and tok_ratio < PACKING_RATIO_MAX
        and explicit_span(BATCH_BUILD)
    ):
        findings.append(
            {
                "family": "batch_packing",
                "ratio": tok_ratio,
                "reason": f"tokens/iteration candidate/baseline={tok_ratio:.3f}",
            }
        )

    queue_ratio = _metric_ratio(profile, ("queue_depth", "waiting_requests"))
    if (
        queue_ratio is not None
        and queue_ratio >= SCHEDULER_RATIO_MIN
        and explicit_span(QUEUE_WAIT, ADMISSION)
    ):
        findings.append(
            {
                "family": "queue_admission",
                "ratio": queue_ratio,
                "reason": f"queue depth candidate/baseline={queue_ratio:.3f}",
            }
        )

    for signal in profile.scheduler_signals:
        if canonical_scheduler_family(signal.name) is None or signal.source not in (
            "naive-sidecar",
            "hctracer.marker",
        ):
            continue
        if signal.ratio is None or signal.ratio < SCHEDULER_RATIO_MIN:
            continue
        findings.append(
            {
                "family": signal.family,
                "name": signal.name,
                "ratio": signal.ratio,
                "source": signal.source,
                "reason": f"{signal.name} duration ratio={signal.ratio:.3f}",
            }
        )
    return findings


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

    graph_timing_mode = _value(graph_timing, "evidence_mode")
    graph_launch_time = _value(graph_timing, "graph_launch_us") or 0.0
    graph_replay_time = _value(graph_timing, "graph_replay_device_us") or 0.0
    graph_timing_paired = (
        graph_timing is not None
        and graph_timing_mode == "EXCLUSIVE_UNION_WITH_EXPLICIT_CORRELATION"
        and float(graph_launch_time) > 0.0
        and float(graph_replay_time) > 0.0
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
                f"graph_launch_us={float(graph_launch_time):.3f}",
                f"graph_replay_device_us={float(graph_replay_time):.3f}",
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
                "priority_rule": 2,
                "score": float(graph_replay_time or graph_launch_time or 0.0),
            }
        )
        ranked.append(graph_path)
        seen.add(str(graph_path["path_id"]))

    for finding in scheduler_findings(profile):
        path_id = f"scheduler.{finding['family']}"
        if path_id in seen:
            continue
        seen.add(path_id)
        ranked.append(
            {
                "rank": len(ranked) + 1,
                "path_id": path_id,
                "priority_rule": 1,
                "score": float(finding["ratio"]),
                "reasons": [finding["reason"], "explicit scheduler-layer evidence"],
                "lever": LEVERS[path_id],
            }
        )

    if host_first_fired(profile, wall_ratio) and "runtime.host_graph_command" not in seen:
        seen.add("runtime.host_graph_command")
        ranked.append(
            {
                "rank": len(ranked) + 1,
                "path_id": "runtime.host_graph_command",
                "priority_rule": 3,
                "reasons": [
                    "host runtime/command/gap evidence crossed the documented threshold",
                    "this is not scheduler packing effort",
                ],
                "lever": LEVERS["runtime.host_graph_command"],
            }
        )

    if not ranked:
        path_id = "diagnosis.insufficient_evidence"
        ranked.append(
            {
                "rank": 1,
                "path_id": path_id,
                "priority_rule": 99,
                "reasons": [
                    "no scheduler annotation and no host-first threshold crossed",
                    "collect sidecar/HCTX spans or a longer matched window",
                ],
                "lever": "collect missing signals before selecting a code lever",
            }
        )

    for rank, item in enumerate(ranked, 1):
        item["rank"] = rank
    return {
        "top_path": ranked[0]["path_id"],
        "host_first_fired": host_first_fired(profile, wall_ratio),
        "scheduler_evidence_count": len(scheduler_findings(profile)),
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
