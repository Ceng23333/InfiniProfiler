"""Stable in-memory representation of infiniprofiler-profile-1."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class HostFamily(str, Enum):
    API = "api"
    COMMAND = "command"
    DRIVER = "driver"
    RUNTIME = "runtime"
    GAP = "gap"
    GRAPH = "graph"


class SchedulerFamily(str, Enum):
    STEP_CONTROL = "step_control"
    BATCH_PACKING = "batch_packing"
    QUEUE_ADMISSION = "queue_admission"


@dataclass
class HostSignal:
    family: str
    candidate: float
    baseline: Optional[float] = None
    ratio: Optional[float] = None
    unit: str = "us"
    source: str = "unknown"
    note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class SchedulerSignal:
    family: str
    name: str
    candidate_us: float
    count: int
    baseline_us: Optional[float] = None
    ratio: Optional[float] = None
    source: str = "unknown"
    ranks: List[int] = field(default_factory=list)
    sample: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class EngineMetric:
    name: str
    candidate: Optional[float] = None
    baseline: Optional[float] = None
    ratio: Optional[float] = None
    unit: str = "value"
    source: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class Capture:
    source: str
    mode: str = "attach"
    pid: Optional[int] = None
    ranks: List[int] = field(default_factory=list)
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class ProfileDocument:
    capture: Capture
    host_signals: List[HostSignal] = field(default_factory=list)
    scheduler_signals: List[SchedulerSignal] = field(default_factory=list)
    metrics: List[EngineMetric] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def schema_version(self) -> str:
        return "infiniprofiler-profile-1"

    def host(self, family: str) -> Optional[HostSignal]:
        return next((x for x in self.host_signals if x.family == family), None)

    def scheduler(self, family: str) -> Optional[SchedulerSignal]:
        return next((x for x in self.scheduler_signals if x.family == family), None)

    def metric(self, name: str) -> Optional[EngineMetric]:
        return next((x for x in self.metrics if x.name == name), None)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "capture": self.capture.to_dict(),
            "host_signals": [x.to_dict() for x in self.host_signals],
            "scheduler_signals": [x.to_dict() for x in self.scheduler_signals],
            "metrics": [x.to_dict() for x in self.metrics],
            "warnings": list(self.warnings),
        }

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def safe_ratio(candidate: float, baseline: float) -> Optional[float]:
    if baseline <= 0:
        return None
    return candidate / baseline


def merge_profiles(primary: ProfileDocument, fallback: ProfileDocument) -> ProfileDocument:
    """Merge fallback evidence without replacing primary capture provenance."""
    seen_host = {x.family for x in primary.host_signals}
    seen_sched = {(x.family, x.name, x.source) for x in primary.scheduler_signals}
    seen_metric = {x.name for x in primary.metrics}

    merged = ProfileDocument(capture=primary.capture)
    merged.host_signals.extend(primary.host_signals)
    merged.scheduler_signals.extend(primary.scheduler_signals)
    merged.metrics.extend(primary.metrics)
    merged.warnings.extend(primary.warnings)

    sources = primary.capture.provenance.setdefault("sources", [])
    if fallback.capture.source not in sources:
        sources.append(fallback.capture.source)

    for signal in fallback.host_signals:
        if signal.family not in seen_host:
            merged.host_signals.append(signal)
    for signal in fallback.scheduler_signals:
        key = (signal.family, signal.name, signal.source)
        if key not in seen_sched:
            merged.scheduler_signals.append(signal)
    for metric in fallback.metrics:
        if metric.name not in seen_metric:
            merged.metrics.append(metric)
    for warning in fallback.warnings:
        if warning not in merged.warnings:
            merged.warnings.append(warning)
    return merged
