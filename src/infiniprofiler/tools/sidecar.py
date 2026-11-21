"""Parse canonical scheduler spans and engine metrics from JSONL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..models import (
    Capture,
    EngineMetric,
    ProfileDocument,
    SchedulerSignal,
    safe_ratio,
)
from ..scheduler import (
    canonical_scheduler_family,
    scheduler_like,
)


STRUCTURED_FIELDS = (
    "iteration",
    "queue_depth",
    "batch_size",
    "tokens_per_iteration",
    "acceptance_count",
    "preempt_count",
)


def _duration_us(item: Dict[str, Any]) -> Optional[float]:
    if "duration_us" in item:
        try:
            return max(0.0, float(item["duration_us"]))
        except (TypeError, ValueError):
            return None
    if "duration_ns" in item:
        try:
            return max(0.0, float(item["duration_ns"]) / 1000.0)
        except (TypeError, ValueError):
            return None
    try:
        start = float(item["start_ns"])
        end = float(item["end_ns"])
    except (KeyError, TypeError, ValueError):
        return None
    return max(0.0, (end - start) / 1000.0)


def _number(item: Dict[str, Any], key: str) -> Optional[float]:
    value = item.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _read_jsonl(path: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    warnings: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                warnings.append(f"invalid_jsonl_line:{line_number}:{exc.msg}")
                continue
            if isinstance(row, dict):
                rows.append(row)
            else:
                warnings.append(f"non_object_jsonl_line:{line_number}")
    return rows, warnings


def _aggregate(
    rows: List[Dict[str, Any]],
) -> Tuple[
    Dict[tuple, Dict[str, Any]],
    Dict[str, Dict[str, Any]],
    List[str],
    set,
]:
    scheduler: Dict[tuple, Dict[str, Any]] = {}
    metrics: Dict[str, Dict[str, Any]] = {}
    warnings: List[str] = []
    window_ids: set = set()

    for line_number, row in enumerate(rows, 1):
        kind = str(row.get("type") or "")
        window_id = row.get("window_id")
        if window_id is not None:
            window_ids.add(str(window_id))

        if kind == "span":
            name = str(row.get("name") or "")
            duration = _duration_us(row)
            family = canonical_scheduler_family(name)
            if not name or duration is None or family is None:
                if scheduler_like(name):
                    warnings.append(
                        f"unknown_scheduler_span_ignored:{line_number}:{name}"
                    )
                else:
                    warnings.append(f"non_scheduler_span_ignored:{line_number}:{name}")
                continue
            if window_id is None:
                warnings.append(f"span_missing_window_id:{line_number}:{name}")
            key = (family, name)
            item = scheduler.setdefault(
                key,
                {
                    "total_us": 0.0,
                    "count": 0,
                    "max_us": 0.0,
                    "ranks": set(),
                    "structured": {},
                },
            )
            item["total_us"] += duration
            item["count"] += 1
            item["max_us"] = max(float(item["max_us"]), duration)
            rank = _number(row, "rank")
            if rank is not None:
                item["ranks"].add(int(rank))
            for field_name in STRUCTURED_FIELDS:
                value = _number(row, field_name)
                if value is None:
                    continue
                stats = item["structured"].setdefault(
                    field_name,
                    {
                        "sum": 0.0,
                        "count": 0,
                        "min": value,
                        "max": value,
                        "last": value,
                    },
                )
                stats["sum"] += value
                stats["count"] += 1
                stats["min"] = min(float(stats["min"]), value)
                stats["max"] = max(float(stats["max"]), value)
                stats["last"] = value
        elif kind == "metric":
            name = str(row.get("name") or "")
            value = _number(row, "value")
            if not name or value is None:
                warnings.append(f"invalid_metric_line:{line_number}")
                continue
            item = metrics.setdefault(
                name,
                {"sum": 0.0, "count": 0, "max": value},
            )
            item["sum"] += value
            item["count"] += 1
            item["max"] = max(float(item["max"]), value)
        elif kind in ("start", "stop"):
            # Window delimiters may carry identity; span rows remain the source
            # of scheduler facts.
            pass
        else:
            warnings.append(f"unknown_record_type:{line_number}:{kind}")
    return scheduler, metrics, warnings, window_ids


def normalize(
    input_path: Path,
    *,
    baseline_path: Optional[Path] = None,
    mode: str = "sidecar",
) -> ProfileDocument:
    rows, warnings = _read_jsonl(input_path)
    scheduler, metrics, parse_warnings, window_ids = _aggregate(rows)
    warnings.extend(parse_warnings)

    baseline_scheduler: Dict[str, Dict[str, Any]] = {}
    baseline_metrics: Dict[str, Dict[str, Any]] = {}
    baseline_windows: set = set()
    if baseline_path is not None:
        baseline_rows, baseline_warnings = _read_jsonl(baseline_path)
        baseline_scheduler_raw, baseline_metrics, baseline_parse_warnings, baseline_windows = _aggregate(
            baseline_rows
        )
        warnings.extend("baseline:" + item for item in baseline_warnings)
        warnings.extend("baseline:" + item for item in baseline_parse_warnings)
        if len(baseline_windows) != 1:
            warnings.append("baseline:sidecar_window_not_matched")
        for (family, name), item in baseline_scheduler_raw.items():
            baseline_scheduler.setdefault(family, {}).setdefault(name, item)

    scheduler_signals: List[SchedulerSignal] = []
    for (family, name), item in sorted(scheduler.items()):
        candidate = float(item["total_us"])
        baseline_names = baseline_scheduler.get(family, {})
        baseline_item = baseline_names.get(name)
        baseline = float(baseline_item["total_us"]) if baseline_item else None
        ranks: List[int] = sorted(item["ranks"])
        scheduler_signals.append(
            SchedulerSignal(
                family=family,
                name=name,
                candidate_us=candidate,
                count=int(item["count"]),
                baseline_us=baseline,
                ratio=safe_ratio(candidate, baseline) if baseline else None,
                source="naive-sidecar",
                ranks=ranks,
                sample={
                    "avg_us": candidate / int(item["count"]),
                    "max_us": item["max_us"],
                    "structured": {
                        field: {
                            "avg": stats["sum"] / stats["count"],
                            "min": stats["min"],
                            "max": stats["max"],
                            "last": stats["last"],
                        }
                        for field, stats in item["structured"].items()
                    },
                },
            )
        )

    metric_signals: List[EngineMetric] = []
    for name, item in sorted(metrics.items()):
        candidate = float(item["sum"]) / int(item["count"])
        baseline_item = baseline_metrics.get(name)
        baseline = (
            float(baseline_item["sum"]) / int(baseline_item["count"])
            if baseline_item
            else None
        )
        metric_signals.append(
            EngineMetric(
                name=name,
                candidate=candidate,
                baseline=baseline,
                ratio=safe_ratio(candidate, baseline) if baseline else None,
                source="naive-sidecar",
            )
        )

    if not scheduler_signals:
        warnings.append("no_scheduler_spans: no canonical InfiniLM spans were accepted")
    matched_window = bool(scheduler_signals) and len(window_ids) == 1
    if not matched_window:
        warnings.append("sidecar_window_not_matched")

    provenance: Dict[str, Any] = {
        "input": str(input_path),
        "row_count": len(rows),
        "window_id": next(iter(window_ids)) if len(window_ids) == 1 else None,
        "window_matched": matched_window,
    }
    if baseline_path is not None:
        baseline_window_id = next(iter(baseline_windows)) if len(baseline_windows) == 1 else None
        windows_matched = (
            matched_window
            and len(baseline_windows) == 1
            and window_ids == baseline_windows
        )
        provenance.update(
            {
                "baseline_window_id": baseline_window_id,
                "windows_matched": windows_matched,
            }
        )
        if not windows_matched:
            warnings.append("sidecar_windows_not_matched")

    return ProfileDocument(
        capture=Capture(
            source="naive-sidecar",
            mode=mode,
            provenance=provenance,
        ),
        scheduler_signals=scheduler_signals,
        metrics=metric_signals,
        warnings=warnings,
    )
