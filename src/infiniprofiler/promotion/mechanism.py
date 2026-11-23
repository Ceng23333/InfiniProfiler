"""Build promotion-request-2 and evaluate declared mechanism movement."""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..contracts import (
    REQUEST_SCHEMA,
    LeverContract,
    parse_lever_contract,
    validate_promotion_request_v2,
)
from ..models import ProfileDocument
from .ranker import primary_wall_ratio, rank_paths


class MechanismError(ValueError):
    """Raised when a promotion-request-2 bridge would be unsafe or inconsistent."""


@dataclass
class MechanismEvaluation:
    status: str
    before_profile_ref: str
    after_profile_ref: str
    results: List[Dict[str, Any]]
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _section(document: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in document and isinstance(document[name], Mapping):
            return document[name]
    return None


def _timing(document: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    direct = _section(document, "timing_attribution", "timing")
    if direct is not None:
        return direct
    if "graph_launch_us" in document or "graph_replay_device_us" in document:
        return document
    return None


def _rank_views(document: Mapping[str, Any]) -> Optional[List[Any]]:
    for name in ("ranks", "rank_views"):
        value = document.get(name)
        if isinstance(value, list):
            return value
    return None


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _timing_value(timing: Mapping[str, Any], name: str) -> Optional[float]:
    return _number(timing.get(name))


def _rank_value(document: Mapping[str, Any], name: str) -> Optional[float]:
    views = _rank_views(document)
    if not views:
        return None
    values = [
        _number(item.get(name))
        for item in views
        if isinstance(item, Mapping) and _number(item.get(name)) is not None
    ]
    values = [value for value in values if value is not None]
    return float(statistics.median(values)) if values else None


def _metric_value(document: Mapping[str, Any], name: str) -> Optional[float]:
    metrics = document.get("metrics")
    if not isinstance(metrics, list):
        return None
    for item in metrics:
        if not isinstance(item, Mapping) or item.get("name") != name:
            continue
        if name.endswith("_ratio") or item.get("ratio") is not None and name.endswith("ratio"):
            return _number(item.get("ratio"))
        return _number(item.get("candidate"))
    return None


def signal_value(document: Mapping[str, Any], signal_name: str) -> Tuple[Optional[float], str]:
    """Resolve a timing, rank, or engine-metric signal from an artifact."""
    if signal_name.startswith("timing."):
        timing = _timing(document)
        return (
            _timing_value(timing, signal_name.split(".", 1)[1]) if timing else None,
            "timing",
        )
    if signal_name.startswith("rank."):
        return _rank_value(document, signal_name.split(".", 1)[1]), "rank"
    if signal_name.startswith("metric."):
        return _metric_value(document, signal_name.split(".", 1)[1]), "metric"
    if signal_name.startswith("scheduler."):
        return None, "missing"

    timing = _timing(document)
    if timing is not None and _timing_value(timing, signal_name) is not None:
        return _timing_value(timing, signal_name), "timing"
    rank_value = _rank_value(document, signal_name)
    if rank_value is not None:
        return rank_value, "rank"
    metric_value = _metric_value(document, signal_name)
    if metric_value is not None:
        return metric_value, "metric"
    return None, "missing"


def _identity(document: Mapping[str, Any], label: str) -> Dict[str, Any]:
    window = _mapping(document.get("window"))
    workload = _mapping(document.get("workload"))
    capture = _mapping(document.get("capture"))
    provenance = _mapping(capture.get("provenance"))
    source_map = _mapping(document.get("source_map"))
    return {
        "window_id": (
            window.get("window_id")
            or document.get("window_id")
            or provenance.get("window_id")
        ),
        "campaign_id": workload.get("campaign_id") or document.get("campaign_id"),
        "model": workload.get("model") or document.get("model"),
        "source_root_hash": source_map.get("source_root_hash") or document.get("source_root_hash"),
        "rank_count": len(_rank_views(document) or []),
        "window_matched": (
            window.get("alignment") == "MATCHED"
            or provenance.get("window_matched") is True
            or document.get("boundary_status") == "MATCHED_GRAPH_REPLAY"
        ),
    }


def _identity_inconclusive(
    lever: LeverContract,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> List[str]:
    before_identity = _identity(before, "before")
    after_identity = _identity(after, "after")
    issues: List[str] = []
    for key in ("window_id", "campaign_id", "model", "source_root_hash"):
        if before_identity[key] is None or after_identity[key] is None:
            issues.append(f"missing_{key}")
        elif before_identity[key] != after_identity[key]:
            issues.append(f"mismatched_{key}")
    if before_identity["window_matched"] is not True or after_identity["window_matched"] is not True:
        issues.append("window_not_matched")
    if (
        before_identity["rank_count"] > 0
        and after_identity["rank_count"] > 0
        and before_identity["rank_count"] != after_identity["rank_count"]
    ):
        issues.append("mismatched_rank_count")
    return issues


def _evaluate_signal(
    *,
    signal_name: str,
    direction: str,
    min_ratio: float,
    before_value: Optional[float],
    after_value: Optional[float],
    source: str,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "signal": signal_name,
        "direction": direction,
        "min_ratio": min_ratio,
        "before": before_value,
        "after": after_value,
        "ratio": None,
        "source": source,
        "status": "INCONCLUSIVE",
        "reason": "signal missing",
    }
    if before_value is None or after_value is None:
        return result
    if before_value <= 0:
        result["reason"] = "non-positive before value"
        return result
    ratio = after_value / before_value
    result["ratio"] = ratio
    if direction == "decrease":
        favorable = after_value < before_value
    else:
        favorable = after_value > before_value
    if not favorable:
        result["status"] = "REGRESSED"
        result["reason"] = "mechanism moved in the wrong direction"
        return result
    if direction == "decrease":
        passed = ratio <= min_ratio
    else:
        passed = ratio >= min_ratio
    result["status"] = "PASSED" if passed else "NO_CHANGE"
    result["reason"] = (
        f"after/before={ratio:.3f} {'<=' if direction == 'decrease' else '>='} {min_ratio:.3f}"
        if passed
        else f"after/before={ratio:.3f} did not cross {min_ratio:.3f}"
    )
    return result


def evaluate_mechanism(
    lever_contract: Mapping[str, Any],
    *,
    before: Mapping[str, Any],
    after: Optional[Mapping[str, Any]],
    before_profile_ref: str,
    after_profile_ref: str = "",
) -> MechanismEvaluation:
    """Evaluate every declared signal and classify mechanism movement."""
    lever = parse_lever_contract(lever_contract)
    if after is None:
        return MechanismEvaluation(
            status="NOT_RUN",
            before_profile_ref=before_profile_ref,
            after_profile_ref=after_profile_ref,
            results=[],
            warnings=["after profile not supplied"],
        )

    identity_issues = _identity_inconclusive(lever, before, after)
    results: List[Dict[str, Any]] = []
    warnings: List[str] = []
    if identity_issues:
        warnings.append("identity_issues=" + ",".join(identity_issues))

    for signal in lever.expected_signals:
        before_value, before_source = signal_value(before, signal.name)
        after_value, after_source = signal_value(after, signal.name)
        source = before_source if before_source == after_source else "missing"
        result = _evaluate_signal(
            signal_name=signal.name,
            direction=signal.direction,
            min_ratio=signal.min_ratio,
            before_value=before_value,
            after_value=after_value,
            source=source,
        )
        if identity_issues:
            result["status"] = "INCONCLUSIVE"
            result["reason"] = "identity_issues=" + ",".join(identity_issues)
        if source == "timing":
            before_timing = _timing(before)
            after_timing = _timing(after)
            evidence_modes = [
                item.get("evidence_mode")
                for item in (before_timing, after_timing)
                if item is not None
            ]
            if any(mode != "EXCLUSIVE_UNION_WITH_EXPLICIT_CORRELATION" for mode in evidence_modes):
                result["status"] = "INCONCLUSIVE"
                result["reason"] = "timing evidence is not explicitly correlated"
        if source == "rank" and (
            _mapping(before).get("boundary_status") != "MATCHED_GRAPH_REPLAY"
            or _mapping(after).get("boundary_status") != "MATCHED_GRAPH_REPLAY"
        ):
            result["status"] = "INCONCLUSIVE"
            result["reason"] = "rank evidence is not matched at graph-replay boundaries"
        results.append(result)

    statuses = [result["status"] for result in results]
    if identity_issues or "INCONCLUSIVE" in statuses:
        status = "INCONCLUSIVE"
    elif statuses and all(item == "PASSED" for item in statuses):
        status = "MOVED"
    elif "REGRESSED" in statuses:
        status = "REGRESSED"
    else:
        status = "NO_CHANGE"
    return MechanismEvaluation(
        status=status,
        before_profile_ref=before_profile_ref,
        after_profile_ref=after_profile_ref,
        results=results,
        warnings=warnings,
    )


def build_promotion_request_v2(
    profile: ProfileDocument,
    *,
    wall_compare: Mapping[str, Any],
    lever_contract: Mapping[str, Any],
    window_id: str,
    campaign_id: str,
    model: str,
    host: str,
    device_track: str,
    baseline_pin: str,
    candidate_pin: str,
    iteration: str,
    profile_ref: str,
    before_profile_ref: str,
    after_profile_ref: str = "",
    before: Optional[Mapping[str, Any]] = None,
    after: Optional[Mapping[str, Any]] = None,
    baseline_backend: str = "baseline",
    candidate_backend: str = "candidate",
    exhausted: Optional[Sequence[str]] = None,
    graph_timing: Optional[Any] = None,
    graph_source: Optional[Any] = None,
    graph_ranks: Optional[Any] = None,
) -> Dict[str, Any]:
    """Build the deterministic AutoHelix v2 measure request and stop before promote."""
    lever = parse_lever_contract(lever_contract)
    if profile.capture.source == "naive-sidecar":
        provenance = profile.capture.provenance
        if provenance.get("window_matched") is not True:
            raise MechanismError("naive-sidecar profile has no matched window; request rejected")
        if provenance.get("windows_matched") is False:
            raise MechanismError("naive-sidecar windows do not match; request rejected")

    wall_ratio = primary_wall_ratio(wall_compare)
    timing = graph_timing
    if timing is None and profile.timing_attribution is not None:
        timing = profile.timing_attribution.to_dict()
    backlog = rank_paths(
        profile,
        wall_ratio=wall_ratio,
        exhausted=exhausted,
        graph_timing=timing,
        graph_source=graph_source,
        graph_ranks=graph_ranks,
    )
    if backlog["top_path"] != lever.top_path:
        raise MechanismError(
            f"lever_contract.top_path={lever.top_path} does not match observed top_path={backlog['top_path']}"
        )

    mechanism = evaluate_mechanism(
        lever_contract,
        before=before or {},
        after=after,
        before_profile_ref=before_profile_ref,
        after_profile_ref=after_profile_ref,
    )
    payload = {
        "schema_version": REQUEST_SCHEMA,
        "controller": "autohelix",
        "stop_before_promote": True,
        "campaign": {
            "campaign_id": campaign_id,
            "model": model,
            "host": host,
            "device_track": device_track,
            "baseline_backend": baseline_backend,
            "candidate_backend": candidate_backend,
            "baseline_pin": baseline_pin,
            "candidate_pin": candidate_pin,
            "iteration": iteration,
        },
        "observation": {
            "profile_ref": profile_ref,
            "profile_schema_version": profile.schema_version,
            "profile_source": profile.capture.source,
            "profile_mode": profile.capture.mode,
            "profile_warnings": profile.warnings,
            "window": {"window_id": window_id, "alignment": "MATCHED"},
            "wall_ratio": wall_ratio,
            "wall_compare": dict(wall_compare),
            "timing_attribution": timing,
            "graph_timing": timing,
            "graph_source": graph_source,
            "graph_ranks": graph_ranks,
        },
        "gate": {
            "metric": "unprofiled_primary_wall",
            "surpass_threshold": lever.wall_ratio_gt,
            "profiled_result_can_promote": False,
        },
        "diagnosis": backlog,
        "lever_contract": lever_contract,
        "promote": {
            "one_path": lever.top_path,
            "lever_hint": lever.lever_id,
            "may_apply": False,
            "must_reobserve_unprofiled": True,
        },
        "mechanism_check": mechanism.to_dict(),
    }
    validate_promotion_request_v2(payload)
    return payload


def apply_mechanism_evaluation(
    request: Mapping[str, Any],
    *,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    before_profile_ref: str,
    after_profile_ref: str,
) -> Dict[str, Any]:
    """Evaluate a post-rebuild mechanism and revalidate the v2 request."""
    lever_contract = request.get("lever_contract")
    if not isinstance(lever_contract, Mapping):
        raise MechanismError("request has no lever_contract")
    mechanism = evaluate_mechanism(
        lever_contract,
        before=before,
        after=after,
        before_profile_ref=before_profile_ref,
        after_profile_ref=after_profile_ref,
    )
    payload = dict(request)
    payload["mechanism_check"] = mechanism.to_dict()
    validate_promotion_request_v2(payload)
    return payload
