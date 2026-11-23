"""Optional L3 engine-range provenance (NVTX / sidecar spans).

Ranges are never promotion contracts and never invent packing from CUDA gaps.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


def _duration_us(item: Mapping[str, Any]) -> Optional[float]:
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


def aggregate_engine_ranges(
    rows: Iterable[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Aggregate free-form span rows into provenance-safe summaries."""
    buckets: Dict[str, Dict[str, Any]] = {}
    warnings: List[str] = []
    for index, row in enumerate(rows, 1):
        if str(row.get("type") or "") != "span":
            continue
        name = str(row.get("name") or "").strip()
        duration = _duration_us(row)
        if not name or duration is None:
            warnings.append(f"engine_range_ignored:{index}")
            continue
        item = buckets.setdefault(
            name,
            {"name": name, "total_us": 0.0, "count": 0, "max_us": 0.0},
        )
        item["total_us"] += duration
        item["count"] += 1
        item["max_us"] = max(float(item["max_us"]), duration)
    ranges = [
        {
            "name": name,
            "total_us": float(item["total_us"]),
            "count": int(item["count"]),
            "avg_us": float(item["total_us"]) / int(item["count"]),
            "max_us": float(item["max_us"]),
            "evidence": "optional_engine_range_provenance",
        }
        for name, item in sorted(buckets.items())
    ]
    return ranges, warnings


def provenance_payload(ranges: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Wrap aggregated ranges for capture.provenance (not a rank contract)."""
    return {
        "engine_ranges": ranges,
        "engine_ranges_contract": "provenance_only_not_scheduler_signals",
    }
