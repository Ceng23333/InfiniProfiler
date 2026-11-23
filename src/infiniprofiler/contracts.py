"""Validation contracts for promotion-grade profile and request v2 data."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .models import Capture, EngineMetric, HostSignal


PROFILE_SCHEMA = "infiniprofiler-profile-2"
REQUEST_SCHEMA = "promotion-request-2"
DEFAULT_START_SKEW_MS = 50.0
DEFAULT_DURATION_RATIO_RANGE = (0.95, 1.05)
WINDOW_ALIGNED = "MATCHED"
CORRELATION_STATES = {"EXPLICIT", "UNKNOWN"}
MECHANISM_STATES = {"NOT_RUN", "MOVED", "NO_CHANGE", "REGRESSED", "INCONCLUSIVE"}
SIGNAL_DIRECTIONS = {"increase", "decrease"}


class ContractValidationError(ValueError):
    """Raised when a v2 artifact cannot enter the promotion pipeline."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = list(errors)
        joined = "; ".join(self.errors)
        super().__init__(f"contract validation failed: {joined}")


@dataclass
class WindowAlignment:
    window_id: str
    wall_compare_ref: str
    wall_workload_id: str
    client_started_epoch_ns: int
    client_ended_epoch_ns: int
    warmup_excluded: bool
    start_skew_ms: float
    duration_ratio: float
    alignment: str


@dataclass
class WorkloadIdentity:
    campaign_id: str
    model: str
    backend_baseline_pin: str
    backend_candidate_pin: str
    tensor_parallel: int
    concurrency: int
    prompt_lengths: List[int]
    output_lengths: List[int]
    batch_size: Optional[int]
    clock_policy: str
    profiles_source_hash: str


@dataclass
class RankView:
    rank: int
    pid: int
    gpu_busy_us: float
    start_epoch_ns: int
    end_epoch_ns: int
    start_skew_us: float
    last_finish_skew_us: float
    straggler_wait_us: float
    tp_imbalance_ratio: float
    correlation_status: str
    stream: Optional[str] = None
    tid: Optional[int] = None
    granularity: str = "capture_window"
    boundary_count: int = 0
    boundary_status: str = "UNMATCHED"


@dataclass
class TimingAttribution:
    evidence_mode: str
    api_submit_us: float
    gpu_execution_us: float
    launch_latency_us: float
    sync_wait_us: float
    graph_launch_us: float
    graph_replay_device_us: float
    cpu_gap_before_us: float
    cpu_gap_after_us: float
    unpaired_event_count: int
    paired_event_count: int = 0


@dataclass
class SourceMap:
    source_pin: str
    source_root_hash: str
    symbols: List[Dict[str, Any]]
    providers: List[Dict[str, Any]]
    source_root: Optional[str] = None
    scanned_file_count: int = 0
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    rules: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ProfileV2:
    capture: Capture
    window: WindowAlignment
    workload: WorkloadIdentity
    ranks: List[RankView]
    timing: TimingAttribution
    source_map: SourceMap
    host_signals: List[HostSignal]
    metrics: List[EngineMetric]
    warnings: List[str]


@dataclass
class ExpectedSignal:
    name: str
    direction: str
    min_ratio: float


@dataclass
class LeverContract:
    lever_id: str
    top_path: str
    expected_signals: List[ExpectedSignal]
    source_files: List[str]
    deployment_files: List[str]
    correctness_gate: str
    rollback: str
    wall_ratio_gt: float


@dataclass
class MechanismCheck:
    status: str
    before_profile_ref: str
    after_profile_ref: str
    results: List[Dict[str, Any]]


def _object(value: Any, errors: List[str], path: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        errors.append(f"{path} must be an object")
        return {}
    return value


def _required(document: Mapping[str, Any], keys: Iterable[str], errors: List[str], path: str) -> None:
    for key in keys:
        if key not in document or document[key] in (None, ""):
            errors.append(f"{path}.{key} is required")


def _nonnegative(value: Any, errors: List[str], path: str) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"{path} must be a number")
        return None
    if value < 0:
        errors.append(f"{path} must be non-negative")
        return None
    return float(value)


def _integer(value: Any, errors: List[str], path: str, *, minimum: Optional[int] = None) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int):
        errors.append(f"{path} must be an integer")
        return None
    if minimum is not None and value < minimum:
        errors.append(f"{path} must be >= {minimum}")
        return None
    return int(value)


def _string_list(value: Any, errors: List[str], path: str) -> List[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        errors.append(f"{path} must be a list of non-empty strings")
        return []
    return list(value)


def _parse_window(raw: Dict[str, Any], errors: List[str]) -> Optional[WindowAlignment]:
    path = "window"
    _required(
        raw,
        (
            "window_id",
            "wall_compare_ref",
            "wall_workload_id",
            "client_started_epoch_ns",
            "client_ended_epoch_ns",
            "warmup_excluded",
            "start_skew_ms",
            "duration_ratio",
            "alignment",
        ),
        errors,
        path,
    )
    started = raw.get("client_started_epoch_ns")
    ended = raw.get("client_ended_epoch_ns")
    if isinstance(started, int) and not isinstance(started, bool) and isinstance(ended, int) and not isinstance(ended, bool):
        if ended <= started:
            errors.append(f"{path}.client_ended_epoch_ns must be after start")
    skew = _nonnegative(raw.get("start_skew_ms"), errors, f"{path}.start_skew_ms")
    duration = _nonnegative(raw.get("duration_ratio"), errors, f"{path}.duration_ratio")
    if raw.get("alignment") != WINDOW_ALIGNED:
        errors.append(f"{path}.alignment must be {WINDOW_ALIGNED}")
    if raw.get("warmup_excluded") is not True:
        errors.append(f"{path}.warmup_excluded must be true")
    if skew is not None and skew > DEFAULT_START_SKEW_MS:
        errors.append(f"{path}.start_skew_ms exceeds {DEFAULT_START_SKEW_MS}")
    if duration is not None and not (
        DEFAULT_DURATION_RATIO_RANGE[0] <= duration <= DEFAULT_DURATION_RATIO_RANGE[1]
    ):
        errors.append(
            f"{path}.duration_ratio must be between {DEFAULT_DURATION_RATIO_RANGE[0]} "
            f"and {DEFAULT_DURATION_RATIO_RANGE[1]}"
        )
    if errors:
        return None
    return WindowAlignment(**raw)


def _parse_workload(raw: Dict[str, Any], errors: List[str]) -> Optional[WorkloadIdentity]:
    path = "workload"
    _required(
        raw,
        (
            "campaign_id",
            "model",
            "backend_baseline_pin",
            "backend_candidate_pin",
            "tensor_parallel",
            "concurrency",
            "prompt_lengths",
            "output_lengths",
            "clock_policy",
            "profiles_source_hash",
        ),
        errors,
        path,
    )
    _integer(raw.get("tensor_parallel"), errors, f"{path}.tensor_parallel", minimum=1)
    _integer(raw.get("concurrency"), errors, f"{path}.concurrency", minimum=0)
    batch_size = raw.get("batch_size")
    if batch_size is not None:
        _integer(batch_size, errors, f"{path}.batch_size", minimum=1)
    for key in ("prompt_lengths", "output_lengths"):
        lengths = raw.get(key)
        if not isinstance(lengths, list) or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in lengths
        ):
            errors.append(f"{path}.{key} must contain non-negative integers")
    if errors:
        return None
    return WorkloadIdentity(**raw)


def _parse_rank_views(raw: Any, errors: List[str]) -> List[RankView]:
    if not isinstance(raw, list) or not raw:
        errors.append("ranks must be a non-empty array")
        return []
    ranks: List[RankView] = []
    seen = set()
    for index, item in enumerate(raw):
        path = f"ranks[{index}]"
        row = _object(item, errors, path)
        _required(
            row,
            (
                "rank",
                "pid",
                "gpu_busy_us",
                "start_epoch_ns",
                "end_epoch_ns",
                "start_skew_us",
                "last_finish_skew_us",
                "straggler_wait_us",
                "tp_imbalance_ratio",
                "correlation_status",
            ),
            errors,
            path,
        )
        rank = _integer(row.get("rank"), errors, f"{path}.rank", minimum=0)
        if rank is not None:
            if rank in seen:
                errors.append(f"{path}.rank is duplicated")
            seen.add(rank)
        _integer(row.get("pid"), errors, f"{path}.pid", minimum=1)
        started = row.get("start_epoch_ns")
        ended = row.get("end_epoch_ns")
        if isinstance(started, int) and not isinstance(started, bool) and isinstance(ended, int) and not isinstance(ended, bool):
            if ended <= started:
                errors.append(f"{path}.end_epoch_ns must be after start")
        for key in (
            "gpu_busy_us",
            "start_skew_us",
            "last_finish_skew_us",
            "straggler_wait_us",
            "tp_imbalance_ratio",
        ):
            _nonnegative(row.get(key), errors, f"{path}.{key}")
        if row.get("correlation_status") not in CORRELATION_STATES:
            errors.append(f"{path}.correlation_status must be one of {sorted(CORRELATION_STATES)}")
        if row.get("boundary_count", 0) is not None:
            _integer(row.get("boundary_count", 0), errors, f"{path}.boundary_count", minimum=0)
        if row.get("boundary_status", "UNMATCHED") not in (
            "UNMATCHED",
            "MATCHED_GRAPH_REPLAY",
        ):
            errors.append(
                f"{path}.boundary_status must be UNMATCHED or MATCHED_GRAPH_REPLAY"
            )
        if errors:
            continue
        ranks.append(RankView(**row))
    return ranks


def _parse_timing(raw: Dict[str, Any], errors: List[str]) -> Optional[TimingAttribution]:
    path = "timing_attribution"
    required_keys = (
        "evidence_mode",
        "api_submit_us",
        "gpu_execution_us",
        "launch_latency_us",
        "sync_wait_us",
        "graph_launch_us",
        "graph_replay_device_us",
        "cpu_gap_before_us",
        "cpu_gap_after_us",
        "unpaired_event_count",
    )
    _required(raw, required_keys, errors, path)
    for key in required_keys[1:]:
        _nonnegative(raw.get(key), errors, f"{path}.{key}")
    paired = raw.get("paired_event_count", 0)
    _integer(paired, errors, f"{path}.paired_event_count", minimum=0)
    if errors:
        return None
    return TimingAttribution(**raw)


def _parse_source_map(raw: Dict[str, Any], errors: List[str]) -> Optional[SourceMap]:
    path = "source_map"
    _required(raw, ("source_pin", "source_root_hash", "symbols", "providers"), errors, path)
    if not isinstance(raw.get("symbols"), list) or any(not isinstance(item, dict) for item in raw.get("symbols", [])):
        errors.append(f"{path}.symbols must be an array of objects")
    if not isinstance(raw.get("providers"), list) or any(not isinstance(item, dict) for item in raw.get("providers", [])):
        errors.append(f"{path}.providers must be an array of objects")
    if errors:
        return None
    return SourceMap(**raw)


def validate_profile_v2(document: Mapping[str, Any]) -> ProfileV2:
    """Validate a promotion-grade profile and return its typed representation."""
    errors: List[str] = []
    document = _object(document, errors, "profile")
    if not document:
        raise ContractValidationError(errors)
    if document.get("schema_version") != PROFILE_SCHEMA:
        errors.append(f"schema_version must be {PROFILE_SCHEMA}")

    capture_raw = _object(document.get("capture"), errors, "capture")
    try:
        capture = Capture(**capture_raw)
    except TypeError as exc:
        capture = Capture(source="invalid", mode="invalid")
        errors.append(f"capture fields are invalid: {exc}")

    window_raw = _object(document.get("window"), errors, "window")
    workload_raw = _object(document.get("workload"), errors, "workload")
    timing_raw = _object(document.get("timing_attribution"), errors, "timing_attribution")
    source_raw = _object(document.get("source_map"), errors, "source_map")
    window = _parse_window(window_raw, errors)
    workload = _parse_workload(workload_raw, errors)
    ranks = _parse_rank_views(document.get("ranks"), errors)
    timing = _parse_timing(timing_raw, errors)
    source_map = _parse_source_map(source_raw, errors)

    for key, model in (
        ("host_signals", HostSignal),
        ("metrics", EngineMetric),
    ):
        items = document.get(key)
        if items is None:
            errors.append(f"{key} is required")
            continue
        if not isinstance(items, list):
            errors.append(f"{key} must be an array")
            continue
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                errors.append(f"{key}[{index}] must be an object")
                continue
            try:
                model(**item)
            except TypeError as exc:
                errors.append(f"{key}[{index}] fields are invalid: {exc}")

    warnings = document.get("warnings")
    if not isinstance(warnings, list) or any(not isinstance(item, str) for item in warnings):
        errors.append("warnings must be an array of strings")

    if errors:
        raise ContractValidationError(errors)
    assert window is not None
    assert workload is not None
    assert timing is not None
    assert source_map is not None
    return ProfileV2(
        capture=capture,
        window=window,
        workload=workload,
        ranks=ranks,
        timing=timing,
        source_map=source_map,
        host_signals=[HostSignal(**item) for item in document.get("host_signals", [])],
        metrics=[EngineMetric(**item) for item in document.get("metrics", [])],
        warnings=list(document.get("warnings", [])),
    )


def _parse_lever_contract(raw: Dict[str, Any], errors: List[str]) -> Optional[LeverContract]:
    path = "lever_contract"
    row = _object(raw, errors, path)
    _required(
        row,
        (
            "lever_id",
            "top_path",
            "expected_signals",
            "source_files",
            "deployment_files",
            "correctness_gate",
            "rollback",
            "wall_gate",
        ),
        errors,
        path,
    )
    signals_raw = row.get("expected_signals")
    signals: List[ExpectedSignal] = []
    if not isinstance(signals_raw, list) or not signals_raw:
        errors.append(f"{path}.expected_signals must be a non-empty array")
    else:
        for index, signal in enumerate(signals_raw):
            signal_path = f"{path}.expected_signals[{index}]"
            signal_row = _object(signal, errors, signal_path)
            _required(signal_row, ("name", "direction", "min_ratio"), errors, signal_path)
            if signal_row.get("direction") not in SIGNAL_DIRECTIONS:
                errors.append(f"{signal_path}.direction must be one of {sorted(SIGNAL_DIRECTIONS)}")
            ratio = signal_row.get("min_ratio")
            if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or ratio <= 0:
                errors.append(f"{signal_path}.min_ratio must be positive")
            if not errors:
                signals.append(ExpectedSignal(**signal_row))

    source_files = _string_list(row.get("source_files"), errors, f"{path}.source_files")
    deployment_files = _string_list(row.get("deployment_files"), errors, f"{path}.deployment_files")
    if not source_files and not deployment_files:
        errors.append(f"{path} requires at least one source or deployment file")

    wall_gate = _object(row.get("wall_gate"), errors, f"{path}.wall_gate")
    _required(wall_gate, ("profiled", "ratio_gt"), errors, f"{path}.wall_gate")
    if wall_gate.get("profiled") is not False:
        errors.append(f"{path}.wall_gate.profiled must be false")
    ratio = wall_gate.get("ratio_gt")
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or ratio <= 1.0:
        errors.append(f"{path}.wall_gate.ratio_gt must be greater than 1.0")

    if errors:
        return None
    return LeverContract(
        lever_id=str(row["lever_id"]),
        top_path=str(row["top_path"]),
        expected_signals=signals,
        source_files=source_files,
        deployment_files=deployment_files,
        correctness_gate=str(row["correctness_gate"]),
        rollback=str(row["rollback"]),
        wall_ratio_gt=float(wall_gate["ratio_gt"]),
    )


def _parse_mechanism_check(raw: Any, errors: List[str]) -> MechanismCheck:
    path = "mechanism_check"
    row = _object(raw, errors, path)
    _required(row, ("status",), errors, path)
    if row.get("status") not in MECHANISM_STATES:
        errors.append(f"{path}.status must be one of {sorted(MECHANISM_STATES)}")
    if not isinstance(row.get("results", []), list) or any(
        not isinstance(item, dict) for item in row.get("results", [])
    ):
        errors.append(f"{path}.results must be an array of objects")
    return MechanismCheck(
        status=str(row.get("status", "NOT_RUN")),
        before_profile_ref=str(row.get("before_profile_ref", "")),
        after_profile_ref=str(row.get("after_profile_ref", "")),
        results=list(row.get("results", [])),
    )


def parse_lever_contract(raw: Mapping[str, Any]) -> LeverContract:
    """Parse and validate a promotion lever contract."""
    errors: List[str] = []
    row = _object(raw, errors, "lever_contract")
    lever = _parse_lever_contract(row, errors)
    if errors:
        raise ContractValidationError(errors)
    assert lever is not None
    return lever


def validate_promotion_request_v2(document: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate an AutoHelix v2 request, including its required lever contract."""
    errors: List[str] = []
    row = _object(document, errors, "request")
    if not row:
        raise ContractValidationError(errors)
    if row.get("schema_version") != REQUEST_SCHEMA:
        errors.append(f"schema_version must be {REQUEST_SCHEMA}")
    if row.get("controller") != "autohelix":
        errors.append("controller must be autohelix")
    if row.get("stop_before_promote") is not True:
        errors.append("stop_before_promote must be true")

    campaign = _object(row.get("campaign"), errors, "campaign")
    _required(
        campaign,
        ("campaign_id", "model", "host", "device_track", "baseline_pin", "candidate_pin", "iteration"),
        errors,
        "campaign",
    )
    observation = _object(row.get("observation"), errors, "observation")
    _required(observation, ("profile_ref", "window"), errors, "observation")
    window = _object(observation.get("window"), errors, "observation.window")
    _required(window, ("window_id", "alignment"), errors, "observation.window")
    if window.get("alignment") != WINDOW_ALIGNED:
        errors.append(f"observation.window.alignment must be {WINDOW_ALIGNED}")

    gate = _object(row.get("gate"), errors, "gate")
    _required(gate, ("metric", "surpass_threshold", "profiled_result_can_promote"), errors, "gate")
    if gate.get("metric") != "unprofiled_primary_wall":
        errors.append('gate.metric must be "unprofiled_primary_wall"')
    if gate.get("profiled_result_can_promote") is not False:
        errors.append("gate.profiled_result_can_promote must be false")
    threshold = gate.get("surpass_threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or threshold <= 1.0:
        errors.append("gate.surpass_threshold must be greater than 1.0")

    diagnosis = _object(row.get("diagnosis"), errors, "diagnosis")
    _required(diagnosis, ("top_path", "ranked"), errors, "diagnosis")
    if not isinstance(diagnosis.get("ranked"), list) or not diagnosis.get("ranked"):
        errors.append("diagnosis.ranked must be a non-empty array")

    lever_raw = row.get("lever_contract")
    if lever_raw is None:
        errors.append("lever_contract is required")
        lever = None
    else:
        lever_errors: List[str] = []
        lever = _parse_lever_contract(_object(lever_raw, errors, "lever_contract"), lever_errors)
        errors.extend(lever_errors)

    if lever is not None and diagnosis:
        if lever.top_path != diagnosis.get("top_path"):
            errors.append("lever_contract.top_path must match diagnosis.top_path")
        if lever.top_path != row.get("promote", {}).get("one_path"):
            errors.append("lever_contract.top_path must match promote.one_path")
        gate_threshold = gate.get("surpass_threshold")
        if (
            isinstance(gate_threshold, (int, float))
            and not isinstance(gate_threshold, bool)
            and gate_threshold != lever.wall_ratio_gt
        ):
            errors.append("gate.surpass_threshold must equal lever_contract.wall_gate.ratio_gt")

    promote = _object(row.get("promote"), errors, "promote")
    _required(promote, ("one_path", "may_apply", "must_reobserve_unprofiled"), errors, "promote")
    if promote.get("may_apply") is not False:
        errors.append("promote.may_apply must be false")
    if promote.get("must_reobserve_unprofiled") is not True:
        errors.append("promote.must_reobserve_unprofiled must be true")

    mechanism = _parse_mechanism_check(row.get("mechanism_check", {"status": "NOT_RUN"}), errors)
    if mechanism.status != "NOT_RUN":
        for key in ("before_profile_ref", "after_profile_ref"):
            if not getattr(mechanism, key):
                errors.append(f"mechanism_check.{key} is required when status is {mechanism.status}")
        if not mechanism.results:
            errors.append(f"mechanism_check.results is required when status is {mechanism.status}")
    if lever is not None and len(lever.expected_signals) != len(
        {signal.name for signal in lever.expected_signals}
    ):
        errors.append("lever_contract.expected_signals contains duplicate names")
    if errors:
        raise ContractValidationError(errors)
    return {
        "lever_contract": lever,
        "mechanism_check": mechanism,
    }
