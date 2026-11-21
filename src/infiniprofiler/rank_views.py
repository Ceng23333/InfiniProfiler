"""Aggregate per-rank timeline facts into TP imbalance and straggler views."""

from __future__ import annotations

import json
import os
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .contracts import RankView
from .capture import CaptureError, read_manifest


RANK_VIEWS_SCHEMA = "infiniprofiler-rank-views-1"


class RankViewError(ValueError):
    """Raised when a successful capture cannot produce complete TP facts."""


@dataclass
class RankViews:
    schema_version: str
    status: str
    merge_ref: str
    manifest_ref: str
    window_id: str
    campaign_id: str
    role: str
    expected_rank_count: int
    boundary_status: str
    correlation_status: str
    rank_views: List[RankView]
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RankViewError(f"{path} must be a number")
    return float(value)


def _median(values: List[float]) -> float:
    return float(statistics.median(values))


def _boundary_count(timing: Mapping[str, Any]) -> int:
    boundaries = timing.get("graph_replay_boundaries")
    return len(boundaries) if isinstance(boundaries, list) else 0


def _sorted_boundaries(timing: Mapping[str, Any], rank: int) -> List[Dict[str, Any]]:
    boundaries = timing.get("graph_replay_boundaries")
    if not isinstance(boundaries, list):
        return []
    result: List[Dict[str, Any]] = []
    for index, boundary in enumerate(boundaries):
        if not isinstance(boundary, dict):
            raise RankViewError(f"rank={rank} boundary={index} must be an object")
        required = ("start_us", "end_us", "gpu_busy_us", "correlation_id")
        if any(boundary.get(key) is None for key in required):
            raise RankViewError(f"rank={rank} boundary={index} is incomplete")
        result.append(boundary)
    return sorted(result, key=lambda item: _number(item["start_us"], "boundary.start_us"))


def _base_epoch_ns(
    manifest: Mapping[str, Any],
    timing: Mapping[str, Any],
    minimum_trace_start_us: float,
) -> int:
    started = manifest.get("started_epoch_ns")
    trace_start = timing.get("trace_start_us")
    if isinstance(started, bool) or not isinstance(started, int):
        raise RankViewError("manifest.started_epoch_ns must be an integer")
    start = _number(trace_start, "timing.trace_start_us")
    return started + round((start - minimum_trace_start_us) * 1000)


def build_rank_views(merge: Mapping[str, Any], manifest: Mapping[str, Any]) -> RankViews:
    """Build coarse or graph-boundary-matched TP views from per-rank merges."""
    if merge.get("schema_version") != "infiniprofiler-capture-merge-1":
        raise RankViewError("merge.schema_version must be infiniprofiler-capture-merge-1")
    if merge.get("status") != "COMPLETE":
        raise RankViewError(f"merge.status must be COMPLETE, got {merge.get('status')}")
    manifest_status = manifest.get("status")
    if manifest_status != "SUCCEEDED":
        raise RankViewError(f"manifest.status must be SUCCEEDED, got {manifest_status}")

    ranks = merge.get("ranks")
    if not isinstance(ranks, list) or not ranks:
        raise RankViewError("merge.ranks must be a non-empty array")
    spec = manifest.get("spec", {})
    expected = spec.get("tensor_parallel")
    if isinstance(expected, bool) or not isinstance(expected, int) or len(ranks) != expected:
        raise RankViewError(f"merge must contain exactly {expected} ranks")
    rank_ids = []
    for rank in ranks:
        if not isinstance(rank, dict):
            raise RankViewError("merge rank entries must be objects")
        rank_ids.append(rank.get("rank"))
    if sorted(rank_ids) != list(range(len(ranks))):
        raise RankViewError("merge rank ids must be contiguous and unique")

    warnings: List[str] = []
    timings: Dict[int, Dict[str, Any]] = {}
    minimum_trace_start = float("inf")
    for rank in ranks:
        rank_id = int(rank["rank"])
        timing = rank.get("timing_attribution")
        if not isinstance(timing, dict):
            raise RankViewError(f"rank={rank_id} timing_attribution must be an object")
        if timing.get("trace_start_us") is None or timing.get("trace_end_us") is None:
            raise RankViewError(f"rank={rank_id} has no classified trace span")
        timings[rank_id] = timing
        minimum_trace_start = min(
            minimum_trace_start,
            _number(timing["trace_start_us"], f"rank={rank_id}.trace_start_us"),
        )

    boundary_counts = {
        rank_id: _boundary_count(timings[rank_id]) for rank_id in sorted(timings)
    }
    boundary_status = "MATCHED_GRAPH_REPLAY"
    granularity = "graph_replay_boundary"
    aggregate: Dict[int, Dict[str, float]] = {}

    if len(set(boundary_counts.values())) == 1 and next(iter(boundary_counts.values()), 0) > 0:
        boundary_count = next(iter(boundary_counts.values()))
        per_rank_metrics: Dict[int, List[Dict[str, float]]] = {
            rank_id: [] for rank_id in sorted(timings)
        }
        for index in range(boundary_count):
            metrics: Dict[int, Dict[str, float]] = {}
            for rank_id in sorted(timings):
                boundary = _sorted_boundaries(timings[rank_id], rank_id)[index]
                metrics[rank_id] = {
                    "start_us": _number(boundary["start_us"], "boundary.start_us"),
                    "end_us": _number(boundary["end_us"], "boundary.end_us"),
                    "busy_us": _number(boundary["gpu_busy_us"], "boundary.gpu_busy_us"),
                }
            minimum_start = min(item["start_us"] for item in metrics.values())
            minimum_end = min(item["end_us"] for item in metrics.values())
            maximum_end = max(item["end_us"] for item in metrics.values())
            for rank_id, item in metrics.items():
                per_rank_metrics[rank_id].append(
                    {
                        "start_skew_us": max(0.0, item["start_us"] - minimum_start),
                        "last_finish_skew_us": max(0.0, item["end_us"] - minimum_end),
                        "straggler_wait_us": max(0.0, maximum_end - item["end_us"]),
                        "busy_us": item["busy_us"],
                    }
                )
        for rank_id, values in per_rank_metrics.items():
            aggregate[rank_id] = {
                "gpu_busy_us": sum(item["busy_us"] for item in values),
                "start_skew_us": max(item["start_skew_us"] for item in values),
                "last_finish_skew_us": max(item["last_finish_skew_us"] for item in values),
                "straggler_wait_us": sum(item["straggler_wait_us"] for item in values),
            }
    else:
        boundary_status = "UNMATCHED"
        boundary_count = 0
        granularity = "capture_window"
        warnings.append(
            "graph_replay_boundaries_not_matched_straggler_wait_suppressed"
        )
        for rank_id in sorted(timings):
            timing = timings[rank_id]
            aggregate[rank_id] = {
                "gpu_busy_us": _number(
                    timing.get("gpu_execution_us"), f"rank={rank_id}.gpu_execution_us"
                ),
                "start_skew_us": 0.0,
                "last_finish_skew_us": 0.0,
                "straggler_wait_us": 0.0,
            }
        minimum_trace_end = min(
            _number(timing["trace_end_us"], f"rank={rank_id}.trace_end_us")
            for rank_id, timing in timings.items()
        )
        maximum_trace_end = max(
            _number(timing["trace_end_us"], f"rank={rank_id}.trace_end_us")
            for rank_id, timing in timings.items()
        )
        minimum_trace_start = min(
            _number(timing["trace_start_us"], f"rank={rank_id}.trace_start_us")
            for rank_id, timing in timings.items()
        )
        for rank_id, timing in timings.items():
            aggregate[rank_id]["start_skew_us"] = max(
                0.0,
                _number(timing["trace_start_us"], f"rank={rank_id}.trace_start_us")
                - minimum_trace_start,
            )
            aggregate[rank_id]["last_finish_skew_us"] = max(
                0.0,
                _number(timing["trace_end_us"], f"rank={rank_id}.trace_end_us")
                - minimum_trace_end,
            )
        del maximum_trace_end

    busy_values = [aggregate[rank_id]["gpu_busy_us"] for rank_id in sorted(aggregate)]
    if any(value <= 0 for value in busy_values):
        raise RankViewError("every rank must have positive exclusive GPU busy time")
    median_busy = _median(busy_values)
    maximum_busy = max(busy_values)
    tp_imbalance = maximum_busy / median_busy

    correlation_status = (
        "EXPLICIT"
        if all(timing.get("correlation_status") == "EXPLICIT" for timing in timings.values())
        else "UNKNOWN"
    )
    if correlation_status == "UNKNOWN":
        warnings.append("correlation_unknown_rank_timing_is_coarse")

    rank_views: List[RankView] = []
    for rank in sorted(ranks, key=lambda item: int(item["rank"])):
        rank_id = int(rank["rank"])
        timing = timings[rank_id]
        start_epoch = _base_epoch_ns(manifest, timing, minimum_trace_start)
        end_epoch = start_epoch + round(
            (
                _number(timing["trace_end_us"], f"rank={rank_id}.trace_end_us")
                - _number(timing["trace_start_us"], f"rank={rank_id}.trace_start_us")
            )
            * 1000
        )
        rank_views.append(
            RankView(
                rank=rank_id,
                pid=int(rank["pid"]),
                gpu_busy_us=aggregate[rank_id]["gpu_busy_us"],
                start_epoch_ns=start_epoch,
                end_epoch_ns=end_epoch,
                start_skew_us=aggregate[rank_id]["start_skew_us"],
                last_finish_skew_us=aggregate[rank_id]["last_finish_skew_us"],
                straggler_wait_us=aggregate[rank_id]["straggler_wait_us"],
                tp_imbalance_ratio=tp_imbalance,
                correlation_status=correlation_status,
                granularity=granularity,
                boundary_count=boundary_count,
                boundary_status=boundary_status,
            )
        )

    status = "MATCHED_GRAPH_REPLAY" if boundary_status == "MATCHED_GRAPH_REPLAY" else "COARSE_ONLY"
    return RankViews(
        schema_version=RANK_VIEWS_SCHEMA,
        status=status,
        merge_ref=str(merge.get("manifest", "")),
        manifest_ref=str(merge.get("manifest", "")),
        window_id=str(merge.get("window_id", "")),
        campaign_id=str(spec.get("campaign_id", "")),
        role=str(merge.get("role", "")),
        expected_rank_count=len(ranks),
        boundary_status=boundary_status,
        correlation_status=correlation_status,
        rank_views=rank_views,
        warnings=warnings,
    )


def build_rank_views_from_capture(
    capture_dir: Path,
    output_path: Path,
    *,
    merge: Optional[Mapping[str, Any]] = None,
) -> RankViews:
    merge_path = capture_dir / "capture_merge.json"
    if merge is not None:
        result = build_rank_views(merge, read_manifest(capture_dir / "window_manifest.json"))
        _write_json_atomic(output_path, result.to_dict())
        return result
    try:
        merge = json.loads(merge_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RankViewError(f"cannot read merge {merge_path}: {exc}") from exc
    manifest = read_manifest(capture_dir / "window_manifest.json")
    if not isinstance(merge, dict):
        raise RankViewError(f"merge {merge_path} is not an object")
    result = build_rank_views(merge, manifest)
    _write_json_atomic(output_path, result.to_dict())
    return result
