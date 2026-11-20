"""Rank observed evidence without inventing wall or kernel ratios."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from ..models import ProfileDocument


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
    "kernel.material": "inspect the highest-ratio material device kernel bucket",
}


def _number(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
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
    for key in ("ratio", "WALL_DECODE_RATIO", "wall_ratio"):
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

    tok_ratio = _metric_ratio(
        profile,
        ("tokens_per_iteration", "avg_decoded_tok_per_iter", "tok_per_iter"),
    )
    if tok_ratio is not None and tok_ratio < PACKING_RATIO_MAX:
        findings.append(
            {
                "family": "batch_packing",
                "ratio": tok_ratio,
                "reason": f"tokens/iteration candidate/baseline={tok_ratio:.3f}",
            }
        )

    queue_ratio = _metric_ratio(profile, ("queue_depth", "waiting_requests"))
    if queue_ratio is not None and queue_ratio >= SCHEDULER_RATIO_MIN:
        findings.append(
            {
                "family": "queue_admission",
                "ratio": queue_ratio,
                "reason": f"queue depth candidate/baseline={queue_ratio:.3f}",
            }
        )

    for signal in profile.scheduler_signals:
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
) -> Dict[str, Any]:
    excluded = set(exhausted or [])
    ranked: List[Dict[str, Any]] = []
    seen = set(excluded)

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
                "priority_rule": 2,
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
        "ranked": ranked,
        "exhausted": sorted(excluded),
    }
