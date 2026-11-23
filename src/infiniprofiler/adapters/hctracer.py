"""Chrome-trace normalizer for hcTracer/mcTracer captures."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..models import (
    Capture,
    EngineMetric,
    HostSignal,
    ProfileDocument,
    TimingAttributionFields,
    safe_ratio,
)


API_RE = re.compile(r"^(hc|mc|cuda|cu)[A-Z]", re.IGNORECASE)
LAUNCH_RE = re.compile(r"launch", re.IGNORECASE)
KERNEL_RE = re.compile(r"kernel|gemm|attention|moe|rope|norm", re.IGNORECASE)
MEMCPY_RE = re.compile(r"memcpy|memset|htod|dtoh", re.IGNORECASE)
GRAPH_RE = re.compile(r"graph", re.IGNORECASE)
SCHEDULER_RE = re.compile(
    r"(scheduler|schedule|batch|step|forward|sample|admission)", re.IGNORECASE
)


def load_chrome_trace(path: Path) -> Tuple[Dict[str, Any], List[str]]:
    """Load complete or recoverable hcTracer/mcTracer Chrome JSON."""
    warnings: List[str] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        doc = json.loads(text)
        if isinstance(doc, dict) and isinstance(doc.get("traceEvents"), list):
            return doc, warnings
        raise ValueError("root is not a Chrome trace object")
    except (json.JSONDecodeError, ValueError) as exc:
        first_error = str(exc)

    # hcTracer may be SIGKILLed while stopping an attach capture. Recover every
    # complete object instead of discarding an entire long-running window.
    end = text.rfind("}")
    attempts = 0
    while end > 0 and attempts < 16:
        prefix = text[: end + 1].rstrip().rstrip(",")
        for suffix in ("\n]}\n", "]}\n"):
            try:
                doc = json.loads(prefix + suffix)
            except json.JSONDecodeError:
                continue
            if isinstance(doc, dict) and isinstance(doc.get("traceEvents"), list):
                warnings.append("recovered_truncated_chrome_trace")
                return doc, warnings
        end = text.rfind("}", 0, end)
        attempts += 1
    raise ValueError(f"unrecoverable Chrome trace: {first_error}")


def _events(doc: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for event in doc.get("traceEvents") or []:
        if isinstance(event, dict):
            yield event


def _duration(event: Dict[str, Any]) -> float:
    if event.get("ph") != "X":
        return 0.0
    try:
        return max(0.0, float(event.get("dur", 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _number(event: Dict[str, Any], key: str) -> Optional[float]:
    args = event.get("args")
    candidates: List[Any] = [event.get(key)]
    if isinstance(args, dict):
        candidates.append(args.get(key))
    for value in candidates:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def extract_host_totals_us(doc: Dict[str, Any]) -> Dict[str, float]:
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
    for event in _events(doc):
        name = str(event.get("name") or "")
        duration = _duration(event)
        if duration <= 0:
            continue
        if GRAPH_RE.search(name):
            totals["graph"] += duration
        if LAUNCH_RE.search(name) or str(event.get("cat") or "").lower() == "launch":
            totals["command"] += duration
        if API_RE.search(name):
            totals["api"] += duration
            totals["driver"] += duration
            if not LAUNCH_RE.search(name):
                totals["runtime"] += duration
        if KERNEL_RE.search(name):
            totals["kernel"] += duration
        if MEMCPY_RE.search(name):
            totals["memcpy"] += duration

    busy = totals["kernel"] + totals["memcpy"]
    submitted = totals["api"] + totals["command"]
    totals["gap"] = max(0.0, submitted - busy) if submitted > 0 else 0.0
    return totals


def _host_signals(
    totals: Dict[str, float],
    baseline: Optional[Dict[str, float]],
    *,
    source: str = "hctracer",
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
                source=source,
            )
        )
    return output


def _note_optional_markers(doc: Dict[str, Any], warnings: List[str]) -> None:
    """Record free-form marker names in warnings only; never as a contract."""
    names = sorted(
        {
            str(event.get("name") or "")
            for event in _events(doc)
            if SCHEDULER_RE.search(str(event.get("name") or ""))
            and _duration(event) > 0
        }
    )
    if names:
        warnings.append("optional_marker_names_not_contract:" + ",".join(names[:8]))


def normalize(
    input_path: Path,
    *,
    source_name: str = "hctracer",
    mode: str = "attach",
    pid: Optional[int] = None,
    baseline_path: Optional[Path] = None,
) -> ProfileDocument:
    from ..timeline import attribute_trace

    doc, warnings = load_chrome_trace(input_path)
    totals = extract_host_totals_us(doc)
    baseline_totals = None
    if baseline_path is not None:
        baseline_doc, baseline_warnings = load_chrome_trace(baseline_path)
        baseline_totals = extract_host_totals_us(baseline_doc)
        warnings.extend("baseline:" + item for item in baseline_warnings)

    _note_optional_markers(doc, warnings)

    timeline = attribute_trace(
        input_path,
        database_path=input_path.parent / f"{input_path.stem}.timeline.sqlite3",
    )
    for warning in timeline.warnings:
        if warning not in warnings:
            warnings.append(warning)

    baseline_timing = None
    if baseline_path is not None:
        baseline_timeline = attribute_trace(
            baseline_path,
            database_path=baseline_path.parent / f"{baseline_path.stem}.timeline.sqlite3",
        )
        baseline_timing = TimingAttributionFields(
            evidence_mode=baseline_timeline.evidence_mode,
            api_submit_us=baseline_timeline.api_submit_us,
            gpu_execution_us=baseline_timeline.gpu_execution_us,
            launch_latency_us=baseline_timeline.launch_latency_us,
            sync_wait_us=baseline_timeline.sync_wait_us,
            graph_launch_us=baseline_timeline.graph_launch_us,
            graph_replay_device_us=baseline_timeline.graph_replay_device_us,
            cpu_gap_before_us=baseline_timeline.cpu_gap_before_us,
            cpu_gap_after_us=baseline_timeline.cpu_gap_after_us,
            unpaired_event_count=baseline_timeline.unpaired_event_count,
            paired_event_count=baseline_timeline.paired_event_count,
        )

    timing = TimingAttributionFields(
        evidence_mode=timeline.evidence_mode,
        api_submit_us=timeline.api_submit_us,
        gpu_execution_us=timeline.gpu_execution_us,
        launch_latency_us=timeline.launch_latency_us,
        sync_wait_us=timeline.sync_wait_us,
        graph_launch_us=timeline.graph_launch_us,
        graph_replay_device_us=timeline.graph_replay_device_us,
        cpu_gap_before_us=timeline.cpu_gap_before_us,
        cpu_gap_after_us=timeline.cpu_gap_after_us,
        unpaired_event_count=timeline.unpaired_event_count,
        paired_event_count=timeline.paired_event_count,
    )

    capture = Capture(
        source=source_name,
        mode=mode,
        pid=pid,
        provenance={
            "input": str(input_path),
            "baseline_input": str(baseline_path) if baseline_path else None,
            "event_count": len(doc.get("traceEvents") or []),
            "correlation_status": timeline.correlation_status,
            "timeline_database": timeline.database,
            "baseline_timing_attribution": (
                baseline_timing.to_dict() if baseline_timing is not None else None
            ),
        },
    )
    metrics: List[EngineMetric] = []
    args = doc.get("args") if isinstance(doc.get("args"), dict) else doc
    for name in ("tokens_per_iteration", "batch_size", "queue_depth"):
        value = _number(args, name)
        if value is not None:
            metrics.append(EngineMetric(name=name, candidate=value, source=source_name))

    return ProfileDocument(
        capture=capture,
        host_signals=_host_signals(totals, baseline_totals, source=source_name),
        timing_attribution=timing,
        metrics=metrics,
        warnings=warnings,
    )
