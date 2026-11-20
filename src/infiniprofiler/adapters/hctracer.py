"""Chrome-trace normalizer for hcTracer smoke and attach captures."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from ..models import (
    Capture,
    EngineMetric,
    HostSignal,
    ProfileDocument,
    SchedulerSignal,
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


def extract_scheduler_signals(doc: Dict[str, Any]) -> List[SchedulerSignal]:
    spans: Dict[Tuple[str, str], Dict[str, float]] = {}
    ranks: Dict[Tuple[str, str], Set[int]] = {}
    for event in _events(doc):
        name = str(event.get("name") or "")
        duration = _duration(event)
        if duration <= 0 or not SCHEDULER_RE.search(name):
            continue
        lowered = name.lower()
        family = "step_control"
        if "admission" in lowered:
            family = "queue_admission"
        elif "batch" in lowered:
            family = "batch_packing"
        key = (family, name)
        item = spans.setdefault(
            key,
            {"total_us": 0.0, "count": 0.0, "max_us": 0.0},
        )
        item["total_us"] += duration
        item["count"] += 1
        item["max_us"] = max(item["max_us"], duration)
        rank = _number(event, "rank")
        if rank is not None:
            ranks.setdefault(key, set()).add(int(rank))

    output: List[SchedulerSignal] = []
    for (family, name), item in sorted(spans.items()):
        count = int(item["count"])
        output.append(
            SchedulerSignal(
                family=family,
                name=name,
                candidate_us=item["total_us"],
                count=count,
                source="hctracer.marker",
                ranks=sorted(ranks.get((family, name), set())),
                sample={
                    "avg_us": item["total_us"] / count,
                    "max_us": item["max_us"],
                },
            )
        )
    return output


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
                source="hctracer",
            )
        )
    return output


def normalize(
    input_path: Path,
    *,
    source_name: str = "hctracer",
    mode: str = "attach",
    pid: Optional[int] = None,
    baseline_path: Optional[Path] = None,
) -> ProfileDocument:
    doc, warnings = load_chrome_trace(input_path)
    totals = extract_host_totals_us(doc)
    scheduler = extract_scheduler_signals(doc)
    if not scheduler:
        warnings.append(
            "no_scheduler_annotations: the trace has no HCTX/marker ranges; "
            "host gaps are not scheduler evidence"
        )

    baseline_totals = None
    if baseline_path is not None:
        baseline_doc, baseline_warnings = load_chrome_trace(baseline_path)
        baseline_totals = extract_host_totals_us(baseline_doc)
        warnings.extend("baseline:" + item for item in baseline_warnings)

    capture = Capture(
        source=source_name,
        mode=mode,
        pid=pid,
        provenance={
            "input": str(input_path),
            "baseline_input": str(baseline_path) if baseline_path else None,
            "event_count": len(doc.get("traceEvents") or []),
        },
    )
    metrics: List[EngineMetric] = []
    args = doc.get("args") if isinstance(doc.get("args"), dict) else doc
    for name in ("tokens_per_iteration", "batch_size", "queue_depth"):
        value = _number(args, name)
        if value is not None:
            metrics.append(EngineMetric(name=name, candidate=value, source="hctracer"))

    return ProfileDocument(
        capture=capture,
        host_signals=_host_signals(totals, baseline_totals),
        scheduler_signals=scheduler,
        metrics=metrics,
        warnings=warnings,
    )
