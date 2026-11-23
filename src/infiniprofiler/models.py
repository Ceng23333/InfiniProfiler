"""Timing-first in-memory representation of normalized InfiniProfiler profiles."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


NORMALIZE_SCHEMA = "infiniprofiler-normalize-1"


class HostFamily(str, Enum):
    API = "api"
    COMMAND = "command"
    DRIVER = "driver"
    RUNTIME = "runtime"
    GAP = "gap"
    GRAPH = "graph"


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
class TimingAttributionFields:
    """Exclusive timing buckets shared by normalize and promotion evidence."""

    evidence_mode: str
    api_submit_us: float = 0.0
    gpu_execution_us: float = 0.0
    launch_latency_us: float = 0.0
    sync_wait_us: float = 0.0
    graph_launch_us: float = 0.0
    graph_replay_device_us: float = 0.0
    cpu_gap_before_us: float = 0.0
    cpu_gap_after_us: float = 0.0
    unpaired_event_count: int = 0
    paired_event_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, raw: Dict[str, Any]) -> "TimingAttributionFields":
        return cls(
            evidence_mode=str(raw.get("evidence_mode") or "UNKNOWN"),
            api_submit_us=float(raw.get("api_submit_us") or 0.0),
            gpu_execution_us=float(raw.get("gpu_execution_us") or 0.0),
            launch_latency_us=float(raw.get("launch_latency_us") or 0.0),
            sync_wait_us=float(raw.get("sync_wait_us") or 0.0),
            graph_launch_us=float(raw.get("graph_launch_us") or 0.0),
            graph_replay_device_us=float(raw.get("graph_replay_device_us") or 0.0),
            cpu_gap_before_us=float(raw.get("cpu_gap_before_us") or 0.0),
            cpu_gap_after_us=float(raw.get("cpu_gap_after_us") or 0.0),
            unpaired_event_count=int(raw.get("unpaired_event_count") or 0),
            paired_event_count=int(raw.get("paired_event_count") or 0),
        )


@dataclass
class ProfileDocument:
    capture: Capture
    host_signals: List[HostSignal] = field(default_factory=list)
    timing_attribution: Optional[TimingAttributionFields] = None
    metrics: List[EngineMetric] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def schema_version(self) -> str:
        return NORMALIZE_SCHEMA

    def host(self, family: str) -> Optional[HostSignal]:
        return next((x for x in self.host_signals if x.family == family), None)

    def metric(self, name: str) -> Optional[EngineMetric]:
        return next((x for x in self.metrics if x.name == name), None)

    def timing(self) -> Optional[TimingAttributionFields]:
        return self.timing_attribution

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "capture": self.capture.to_dict(),
            "host_signals": [x.to_dict() for x in self.host_signals],
            "metrics": [x.to_dict() for x in self.metrics],
            "warnings": list(self.warnings),
        }
        if self.timing_attribution is not None:
            payload["timing_attribution"] = self.timing_attribution.to_dict()
        return payload

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
    seen_metric = {x.name for x in primary.metrics}

    merged = ProfileDocument(capture=primary.capture)
    merged.host_signals.extend(primary.host_signals)
    merged.timing_attribution = primary.timing_attribution or fallback.timing_attribution
    merged.metrics.extend(primary.metrics)
    merged.warnings.extend(primary.warnings)

    sources = primary.capture.provenance.setdefault("sources", [])
    if fallback.capture.source not in sources:
        sources.append(fallback.capture.source)

    for signal in fallback.host_signals:
        if signal.family not in seen_host:
            merged.host_signals.append(signal)
    for metric in fallback.metrics:
        if metric.name not in seen_metric:
            merged.metrics.append(metric)
    for warning in fallback.warnings:
        if warning not in merged.warnings:
            merged.warnings.append(warning)
    return merged
