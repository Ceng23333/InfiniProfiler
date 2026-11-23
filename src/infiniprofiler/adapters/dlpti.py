"""Native DLPTI export normalizer (diverge path; not Chrome traceEvents).

``dlpti_tools export --format json`` emits ``activities[]`` with numeric
``kind`` values from ``DLPTI_ActivityKind``. Timed records carry ``start`` /
``end`` timestamps in nanoseconds. ``perfetto-json`` is not used: on SDK
0.10.0 it emits broken nested ``traceEvents`` wrappers.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..models import Capture, HostSignal, ProfileDocument, safe_ratio


# From /usr/local/dlgpu/sdk/include/dlpti/activity.h
KIND_DRIVER_API = 1
KIND_RUNTIME_API = 2
KIND_CU_KERNEL = 20
KIND_TU_KERNEL = 21
KIND_GRAPH = 32
KIND_GRAPH_INSTANCE = 33
KIND_GRAPH_ADD_NODE = 34
KIND_GRAPH_V1 = 35
KIND_COMMAND = 49
KIND_MEMCPY = 50
KIND_MEMCPY4D = 51
KIND_MEMCPY6D = 52
KIND_MEMSET = 53
KIND_MEMSET4D = 54
KIND_MEMSET6D = 55
KIND_CU = 56
KIND_FLUSH = 57
KIND_TU = 58
KIND_MARKER = 82

LAUNCH_RE = re.compile(r"launch", re.IGNORECASE)
GRAPH_NAME_RE = re.compile(r"graph", re.IGNORECASE)

GRAPH_KINDS = {
    KIND_GRAPH,
    KIND_GRAPH_INSTANCE,
    KIND_GRAPH_ADD_NODE,
    KIND_GRAPH_V1,
}
MEMCPY_KINDS = {
    KIND_MEMCPY,
    KIND_MEMCPY4D,
    KIND_MEMCPY6D,
    KIND_MEMSET,
    KIND_MEMSET4D,
    KIND_MEMSET6D,
}
KERNEL_KINDS = {KIND_CU_KERNEL, KIND_TU_KERNEL, KIND_CU, KIND_TU}


def load_native_export(path: Path) -> Tuple[Dict[str, Any], List[str]]:
    """Load a native DLPTI JSON export with an ``activities`` array."""
    warnings: List[str] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    doc = json.loads(text)
    if not isinstance(doc, dict) or not isinstance(doc.get("activities"), list):
        raise ValueError("root is not a DLPTI native export with activities[]")
    return doc, warnings


def _activity_name(act: Dict[str, Any]) -> str:
    for key in ("name", "api_name", "kernel_name", "node_name"):
        value = act.get(key)
        if value not in (None, "", "null"):
            return str(value)
    return f"kind:{act.get('kind')}"


def _duration_us(act: Dict[str, Any]) -> Optional[float]:
    """Return activity duration in microseconds when start/end are present."""
    for start_key, end_key, scale in (
        ("start", "end", 1e-3),  # ns → us
        ("start_time", "end_time", 1.0),
        ("start_us", "end_us", 1.0),
    ):
        if start_key not in act or end_key not in act:
            continue
        try:
            start = float(act[start_key])
            end = float(act[end_key])
        except (TypeError, ValueError):
            continue
        if end < start:
            continue
        delta = end - start
        # Heuristic: values that look like absolute ns timestamps → us
        if scale == 1e-3 and delta > 1e6:
            return delta * scale
        if scale == 1e-3 and start > 1e15:
            return delta * scale
        if "duration" in act:
            try:
                return max(0.0, float(act["duration"]))
            except (TypeError, ValueError):
                pass
        return max(0.0, delta * scale)
    for key in ("duration", "dur", "elapsed_us", "time_us"):
        if key not in act:
            continue
        try:
            val = float(act[key])
        except (TypeError, ValueError):
            continue
        if key.endswith("_us") or (key == "duration" and val > 1000):
            return max(0.0, val if key.endswith("_us") else val / 1000.0)
        return max(0.0, val)
    return None


def extract_host_totals_us(doc: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, Any]]:
    totals = {
        "api": 0.0,
        "command": 0.0,
        "driver": 0.0,
        "runtime": 0.0,
        "kernel": 0.0,
        "memcpy": 0.0,
        "graph": 0.0,
        "gap": 0.0,
    }
    counts = {key: 0 for key in totals}
    timed = 0
    untimed = 0
    for raw in doc.get("activities") or []:
        if not isinstance(raw, dict):
            continue
        try:
            kind = int(raw.get("kind"))
        except (TypeError, ValueError):
            continue
        name = _activity_name(raw)
        dur = _duration_us(raw)
        if dur is None:
            untimed += 1
            # Count proxy (1 us per record) so empty-duration structural
            # records still contribute comparable family mass.
            dur = 1.0
            proxy = True
        else:
            timed += 1
            proxy = False

        if kind == KIND_DRIVER_API:
            totals["driver"] += dur
            totals["api"] += dur
            counts["driver"] += 1
            counts["api"] += 1
            if LAUNCH_RE.search(name):
                totals["command"] += dur
                counts["command"] += 1
        elif kind == KIND_RUNTIME_API:
            totals["runtime"] += dur
            totals["api"] += dur
            counts["runtime"] += 1
            counts["api"] += 1
            if LAUNCH_RE.search(name):
                totals["command"] += dur
                counts["command"] += 1
        elif kind == KIND_COMMAND:
            totals["command"] += dur
            counts["command"] += 1
        elif kind in GRAPH_KINDS or GRAPH_NAME_RE.search(name):
            totals["graph"] += dur
            counts["graph"] += 1
        elif kind in MEMCPY_KINDS or "memcpy" in name.lower() or "memset" in name.lower():
            totals["memcpy"] += dur
            counts["memcpy"] += 1
        elif kind in KERNEL_KINDS:
            totals["kernel"] += dur
            counts["kernel"] += 1
        elif kind == KIND_FLUSH:
            # Flush is device sync overhead; fold into gap numerator later.
            totals["gap"] += dur
            counts["gap"] += 1
        elif proxy:
            # Ignore untimed metadata (device/context/stream) for host families.
            pass

    busy = totals["kernel"] + totals["memcpy"]
    submitted = totals["api"] + totals["command"]
    computed_gap = max(0.0, submitted - busy) if submitted > 0 else 0.0
    totals["gap"] = max(totals["gap"], computed_gap)
    meta = {
        "timed_activities": timed,
        "untimed_activities": untimed,
        "counts": counts,
        "activity_count": timed + untimed,
    }
    return totals, meta


def _host_signals(
    totals: Dict[str, float],
    baseline: Optional[Dict[str, float]],
) -> List[HostSignal]:
    output: List[HostSignal] = []
    for family in ("driver", "runtime", "command", "api", "gap", "graph"):
        candidate = float(totals.get(family, 0.0))
        base = float(baseline.get(family, 0.0)) if baseline else None
        if candidate <= 0 and (base is None or base <= 0):
            continue
        ratio = safe_ratio(candidate, base) if base else None
        output.append(
            HostSignal(
                family=family,
                candidate=candidate,
                baseline=base,
                ratio=ratio,
                source="dlpti",
            )
        )
    return output


def normalize(
    input_path: Path,
    *,
    source_name: str = "dlpti",
    mode: str = "attach",
    pid: Optional[int] = None,
    baseline_path: Optional[Path] = None,
) -> ProfileDocument:
    doc, warnings = load_native_export(input_path)
    totals, meta = extract_host_totals_us(doc)
    baseline_totals = None
    if baseline_path is not None:
        baseline_doc, baseline_warnings = load_native_export(baseline_path)
        baseline_totals, baseline_meta = extract_host_totals_us(baseline_doc)
        warnings.extend("baseline:" + item for item in baseline_warnings)
        meta["baseline"] = baseline_meta

    if meta["timed_activities"] == 0 and meta["activity_count"] > 0:
        warnings.append(
            "dlpti_no_duration_fields: activities lack start/end timing; "
            "host signals use 1us count proxies and are diagnostic only"
        )
    warnings.append(
        "no_scheduler_annotations: DLPTI native export has no InfiniLM "
        "HCTX/marker ranges; host gaps are not scheduler evidence"
    )

    capture = Capture(
        source=source_name,
        mode=mode,
        pid=pid,
        provenance={
            "input": str(input_path),
            "baseline_input": str(baseline_path) if baseline_path else None,
            "activity_count": meta["activity_count"],
            "timed_activities": meta["timed_activities"],
            "vendor_export": "dlpti_tools --format json",
        },
    )
    return ProfileDocument(
        capture=capture,
        host_signals=_host_signals(totals, baseline_totals),
        scheduler_signals=[],
        metrics=[],
        warnings=warnings,
    )
