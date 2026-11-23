"""Parse engine metrics from JSONL sidecars (metrics-only for promotion)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..engine_ranges import aggregate_engine_ranges, provenance_payload
from ..models import (
    Capture,
    EngineMetric,
    ProfileDocument,
    safe_ratio,
)


STRUCTURED_FIELDS = (
    "iteration",
    "queue_depth",
    "batch_size",
    "tokens_per_iteration",
    "acceptance_count",
    "preempt_count",
)


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
) -> Tuple[Dict[str, Dict[str, Any]], List[str], set, List[Dict[str, Any]]]:
    metrics: Dict[str, Dict[str, Any]] = {}
    warnings: List[str] = []
    window_ids: set = set()
    span_rows: List[Dict[str, Any]] = []

    for line_number, row in enumerate(rows, 1):
        kind = str(row.get("type") or "")
        window_id = row.get("window_id")
        if window_id is not None:
            window_ids.add(str(window_id))

        if kind == "span":
            name = str(row.get("name") or "")
            warnings.append(
                f"sidecar_span_provenance_only_not_promotion_evidence:"
                f"{line_number}:{name or 'unnamed'}"
            )
            span_rows.append(row)
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
            for field_name in STRUCTURED_FIELDS:
                field_value = _number(row, field_name)
                if field_value is None:
                    continue
                # Structured fields on metric rows are informational only.
                del field_value
        elif kind in ("start", "stop"):
            pass
        else:
            warnings.append(f"unknown_record_type:{line_number}:{kind}")
    ranges, range_warnings = aggregate_engine_ranges(span_rows)
    warnings.extend(range_warnings)
    return metrics, warnings, window_ids, ranges


def normalize(
    input_path: Path,
    *,
    baseline_path: Optional[Path] = None,
    mode: str = "sidecar",
) -> ProfileDocument:
    rows, warnings = _read_jsonl(input_path)
    metrics, parse_warnings, window_ids, engine_ranges = _aggregate(rows)
    warnings.extend(parse_warnings)

    baseline_metrics: Dict[str, Dict[str, Any]] = {}
    baseline_windows: set = set()
    baseline_ranges: List[Dict[str, Any]] = []
    if baseline_path is not None:
        baseline_rows, baseline_warnings = _read_jsonl(baseline_path)
        (
            baseline_metrics,
            baseline_parse_warnings,
            baseline_windows,
            baseline_ranges,
        ) = _aggregate(baseline_rows)
        warnings.extend("baseline:" + item for item in baseline_warnings)
        warnings.extend("baseline:" + item for item in baseline_parse_warnings)
        if len(baseline_windows) != 1:
            warnings.append("baseline:sidecar_window_not_matched")

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

    if not metric_signals:
        warnings.append("no_engine_metrics: sidecar produced no metric rows")
    matched_window = bool(metric_signals) and len(window_ids) == 1
    if not matched_window:
        warnings.append("sidecar_window_not_matched")

    provenance: Dict[str, Any] = {
        "input": str(input_path),
        "row_count": len(rows),
        "window_id": next(iter(window_ids)) if len(window_ids) == 1 else None,
        "window_matched": matched_window,
    }
    provenance.update(provenance_payload(engine_ranges))
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
                "baseline_engine_ranges": baseline_ranges,
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
        metrics=metric_signals,
        warnings=warnings,
    )
