"""Streaming Chrome-trace timeline attribution for hcTracer captures.

This module is the promotion-grade v2 path. It intentionally avoids loading the
whole ``traceEvents`` array and stages sortable intervals in SQLite instead.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from .adapters.hctracer import API_RE, GRAPH_RE, KERNEL_RE, MEMCPY_RE, SCHEDULER_RE
from .scheduler import canonical_scheduler_family


TIMELINE_SCHEMA = "infiniprofiler-timeline-attribution-1"
DEFAULT_CHUNK_SIZE = 1024 * 1024
DEFAULT_ADJACENCY_MAX_US = 1000.0

SYNC_RE = re.compile(r"synchronize|synchronization|eventrecord|streamwait|wait|barrier", re.IGNORECASE)
LAUNCH_NAME_RE = re.compile(r"launch|replay", re.IGNORECASE)


@dataclass
class TimelineAttribution:
    schema_version: str
    evidence_mode: str
    event_count: int
    classified_event_count: int
    api_submit_us: float
    gpu_execution_us: float
    launch_latency_us: float
    sync_wait_us: float
    graph_launch_us: float
    graph_replay_device_us: float
    cpu_gap_before_us: float
    cpu_gap_after_us: float
    paired_event_count: int
    unpaired_event_count: int
    correlation_status: str
    host_totals_us: Dict[str, float]
    scheduler_signals: List[Dict[str, Any]]
    warnings: List[str]
    database: Optional[str] = None
    trace_start_us: Optional[float] = None
    trace_end_us: Optional[float] = None
    graph_replay_boundaries: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _skip_json_whitespace(text: str, index: int) -> int:
    while index < len(text) and text[index] in " \t\r\n,":
        index += 1
    return index


def _matching_brace(text: str, start: int) -> Optional[int]:
    """Return the matching close brace for a JSON object, respecting strings."""
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    return None


def stream_chrome_events(
    path: Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Iterator[Tuple[Dict[str, Any], List[str]]]:
    """Yield ``(event, new_warnings)`` with a bounded working buffer.

    A truncated hcTracer tail is recoverable. On EOF the final item may have an
    ``event`` of ``None`` and carries any warning for the incomplete tail.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    warnings: List[str] = []
    buffer = ""
    cursor = 0
    at_eof = False

    def refill() -> bool:
        nonlocal buffer, at_eof
        if at_eof:
            return False
        chunk = handle.read(chunk_size)
        if not chunk:
            at_eof = True
            return False
        buffer += chunk
        return True

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        while '"traceEvents"' not in buffer:
            if not refill():
                raise ValueError("traceEvents array not found")
        marker = buffer.find('"traceEvents"')
        cursor = marker + len('"traceEvents"')
        while ":" not in buffer[cursor:]:
            if not refill():
                raise ValueError("traceEvents value not found")
        cursor = buffer.find(":", cursor) + 1
        while "[" not in buffer[cursor:]:
            if not refill():
                raise ValueError("traceEvents array opener not found")
        cursor = buffer.find("[", cursor) + 1
        buffer = buffer[cursor:]
        cursor = 0

        while True:
            cursor = _skip_json_whitespace(buffer, cursor)
            stop = False
            while cursor >= len(buffer):
                if not refill():
                    warnings.append("recovered_truncated_chrome_trace")
                    break
                cursor = _skip_json_whitespace(buffer, cursor)
            else:
                stop = False
            if cursor >= len(buffer):
                stop = True
            if stop:
                break
            if buffer[cursor] == "]":
                if warnings:
                    yield None, warnings
                return

            end = _matching_brace(buffer, cursor)
            if end is None:
                if refill():
                    continue
                warnings.append("recovered_truncated_chrome_trace")
                break
            candidate = buffer[cursor : end + 1]
            try:
                event = json.loads(candidate)
            except json.JSONDecodeError:
                warnings.append("invalid_trace_event_skipped")
            else:
                if isinstance(event, dict):
                    yield event, warnings
                    warnings = []
                else:
                    warnings.append("non_object_trace_event_skipped")
            cursor = end + 1
            if cursor > chunk_size:
                buffer = buffer[cursor:]
                cursor = 0
        if warnings:
            yield None, warnings


def _number(event: Mapping[str, Any], keys: Sequence[str]) -> Optional[float]:
    sources: List[Any] = []
    args = event.get("args")
    if isinstance(args, dict):
        sources.append(args)
    sources.append(event)
    for key in keys:
        for source in sources:
            value = source.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
    return None


def _first_string(mapping: Mapping[str, Any], keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, (str, int)) and not isinstance(value, bool):
            return str(value)
    return None


def _correlation(event: Mapping[str, Any]) -> Optional[str]:
    args = event.get("args")
    candidates: List[Mapping[str, Any]] = []
    if isinstance(args, dict):
        candidates.append(args)
        nested = args.get("correlation")
        if isinstance(nested, dict):
            candidates.append(nested)
    candidates.append(event)
    keys = (
        "correlation",
        "correlation_id",
        "correlationId",
        "correlationID",
        "external_id",
        "externalId",
        "id",
    )
    for source in candidates:
        value = _first_string(source, keys)
        if value is not None:
            return value
    return None


def _is_host(event: Mapping[str, Any], name: str) -> bool:
    category = str(event.get("cat") or "").lower()
    return API_RE.search(name) is not None or category in ("cpu", "host", "runtime", "driver")


def _is_device(event: Mapping[str, Any], name: str) -> bool:
    category = str(event.get("cat") or "").lower()
    if category in ("cpu", "host", "runtime", "driver"):
        return False
    return (
        category in ("kernel", "gpu", "device", "memcpy", "memset")
        or KERNEL_RE.search(name) is not None
        or MEMCPY_RE.search(name) is not None
    )


def classify_event(event: Mapping[str, Any]) -> Optional[str]:
    name = str(event.get("name") or "")
    if event.get("ph") != "X":
        return None
    try:
        if float(event.get("dur", 0.0)) <= 0:
            return None
    except (TypeError, ValueError):
        return None
    lowered = name.lower()
    if SYNC_RE.search(name):
        return "sync"
    if GRAPH_RE.search(name) and _is_device(event, name):
        return "graph_device"
    if GRAPH_RE.search(name) and (_is_host(event, name) or LAUNCH_NAME_RE.search(name)):
        return "graph_launch"
    if _is_device(event, name):
        return "gpu_execution"
    if _is_host(event, name):
        return "api_submit"
    return None


def _interval_total(connection: sqlite3.Connection, category: str) -> float:
    cursor = connection.execute(
        "SELECT start_us, end_us FROM events WHERE category = ? ORDER BY start_us, end_us",
        (category,),
    )
    active_start: Optional[float] = None
    active_end = 0.0
    total = 0.0
    for start, end in cursor:
        start = float(start)
        end = float(end)
        if end <= start:
            continue
        if active_start is None:
            active_start = start
            active_end = end
        elif start <= active_end:
            active_end = max(active_end, end)
        else:
            total += active_end - active_start
            active_start = start
            active_end = end
    if active_start is not None:
        total += active_end - active_start
    return max(0.0, total)


def _gpu_interval_total(connection: sqlite3.Connection) -> float:
    cursor = connection.execute(
        "SELECT start_us, end_us FROM events "
        "WHERE category IN ('gpu_execution', 'graph_device') "
        "ORDER BY start_us, end_us"
    )
    active_start: Optional[float] = None
    active_end = 0.0
    total = 0.0
    for start, end in cursor:
        start = float(start)
        end = float(end)
        if end <= start:
            continue
        if active_start is None:
            active_start = start
            active_end = end
        elif start <= active_end:
            active_end = max(active_end, end)
        else:
            total += active_end - active_start
            active_start = start
            active_end = end
    if active_start is not None:
        total += active_end - active_start
    return max(0.0, total)


def _graph_device_total(connection: sqlite3.Connection) -> float:
    cursor = connection.execute(
        "SELECT start_us, end_us FROM events WHERE category = 'graph_device' "
        "ORDER BY start_us, end_us"
    )
    active_start: Optional[float] = None
    active_end = 0.0
    total = 0.0
    for start, end in cursor:
        if active_start is None:
            active_start = float(start)
            active_end = float(end)
        elif float(start) <= active_end:
            active_end = max(active_end, float(end))
        else:
            total += active_end - active_start
            active_start = float(start)
            active_end = float(end)
    if active_start is not None:
        total += active_end - active_start
    return max(0.0, total)


def _paired_latencies(connection: sqlite3.Connection, *, explicit_only: bool) -> Tuple[float, float, int, int]:
    gpu_cursor = connection.execute(
        "SELECT id, pid, start_us, end_us, correlation_id FROM events "
        "WHERE category IN ('gpu_execution', 'graph_device') ORDER BY start_us, id"
    )
    launch_total = 0.0
    after_total = 0.0
    paired = 0
    unpaired = 0
    adjacency_limit = DEFAULT_ADJACENCY_MAX_US
    for gpu_id, pid, gpu_start, gpu_end, correlation in gpu_cursor:
        paired_row = None
        if correlation is not None:
            paired_row = connection.execute(
                "SELECT end_us, start_us FROM events "
                "WHERE category IN ('api_submit', 'graph_launch') AND pid = ? AND correlation_id = ? "
                "ORDER BY ABS(? - end_us), end_us DESC LIMIT 1",
                    (pid, correlation, float(gpu_start)),
            ).fetchone()
        if paired_row is None and not explicit_only:
            paired_row = connection.execute(
                "SELECT end_us, start_us FROM events "
                "WHERE category IN ('api_submit', 'graph_launch') AND pid = ? AND end_us <= ? "
                "AND end_us >= ? ORDER BY end_us DESC LIMIT 1",
                (pid, float(gpu_start), float(gpu_start) - adjacency_limit),
            ).fetchone()
        if paired_row is None:
            unpaired += 1
            continue
        submit_end = float(paired_row[0])
        next_row = connection.execute(
            "SELECT start_us FROM events "
            "WHERE category = 'api_submit' AND pid = ? AND start_us >= ? "
            "ORDER BY start_us, id LIMIT 1",
            (pid, float(gpu_end)),
        ).fetchone()
        launch_total += max(0.0, float(gpu_start) - submit_end)
        if next_row is not None:
            after_total += max(0.0, float(next_row[0]) - float(gpu_end))
        paired += 1
    return launch_total, after_total, paired, unpaired


def _gpu_interval_total_filtered(
    connection: sqlite3.Connection,
    pid: int,
    correlation_id: str,
) -> float:
    cursor = connection.execute(
        "SELECT start_us, end_us FROM events "
        "WHERE category IN ('gpu_execution', 'graph_device') "
        "AND pid = ? AND correlation_id = ? ORDER BY start_us, end_us",
        (pid, correlation_id),
    )
    active_start: Optional[float] = None
    active_end = 0.0
    total = 0.0
    for start, end in cursor:
        start = float(start)
        end = float(end)
        if end <= start:
            continue
        if active_start is None:
            active_start = start
            active_end = end
        elif start <= active_end:
            active_end = max(active_end, end)
        else:
            total += active_end - active_start
            active_start = start
            active_end = end
    if active_start is not None:
        total += active_end - active_start
    return max(0.0, total)


def _graph_replay_boundaries(connection: sqlite3.Connection) -> List[Dict[str, Any]]:
    launches = connection.execute(
        "SELECT pid, correlation_id, MIN(start_us), MAX(end_us), COUNT(*) "
        "FROM events WHERE category = 'graph_launch' AND correlation_id IS NOT NULL "
        "GROUP BY pid, correlation_id ORDER BY MIN(start_us), pid, correlation_id"
    ).fetchall()
    boundaries: List[Dict[str, Any]] = []
    for pid, correlation, start, end, launch_count in launches:
        devices = connection.execute(
            "SELECT start_us, end_us FROM events "
            "WHERE category = 'graph_device' AND pid = ? AND correlation_id = ? "
            "ORDER BY start_us, end_us",
            (pid, correlation),
        ).fetchall()
        if not devices:
            continue
        device_start = min(float(row[0]) for row in devices)
        device_end = max(float(row[1]) for row in devices)
        boundary_start = min(float(start), device_start)
        boundary_end = max(float(end), device_end)
        boundaries.append(
            {
                "pid": int(pid),
                "correlation_id": str(correlation),
                "launch_count": int(launch_count),
                "device_event_count": len(devices),
                "start_us": boundary_start,
                "end_us": boundary_end,
                "gpu_busy_us": _gpu_interval_total_filtered(connection, int(pid), str(correlation)),
            }
        )
    return boundaries


def _scheduler_signal(
    key: Tuple[str, str],
    aggregate: Dict[str, Any],
    ranks: set,
) -> Dict[str, Any]:
    family, name = key
    count = int(aggregate["count"])
    total = float(aggregate["total_us"])
    return {
        "family": family,
        "name": name,
        "candidate_us": total,
        "count": count,
        "source": "hctracer.marker",
        "ranks": sorted(ranks),
        "sample": {
            "avg_us": total / count,
            "max_us": aggregate["max_us"],
        },
    }


def attribute_trace(
    input_path: Path,
    *,
    database_path: Optional[Path] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> TimelineAttribution:
    """Stream one hcTracer Chrome trace and produce exclusive timing facts."""
    database_path = database_path or input_path.parent / "infiniprofiler_timeline.sqlite3"
    database_path.parent.mkdir(parents=True, exist_ok=True)
    if database_path.exists():
        database_path.unlink()
    connection = sqlite3.connect(str(database_path))
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(
        """
        CREATE TABLE events (
          id INTEGER PRIMARY KEY,
          category TEXT NOT NULL,
          start_us REAL NOT NULL,
          end_us REAL NOT NULL,
          duration_us REAL NOT NULL,
          pid INTEGER,
          tid INTEGER,
          stream TEXT,
          correlation_id TEXT,
          name TEXT NOT NULL,
          is_graph INTEGER NOT NULL
        )
        """
    )
    connection.execute("CREATE INDEX events_category_time ON events(category, start_us, end_us)")
    connection.execute("CREATE INDEX events_pairing ON events(category, pid, correlation_id, end_us)")
    connection.execute("CREATE INDEX events_next_submit ON events(category, pid, start_us)")

    event_count = 0
    classified = 0
    warnings: List[str] = []
    scheduler: Dict[Tuple[str, str], Dict[str, Any]] = {}
    scheduler_ranks: Dict[Tuple[str, str], set] = {}
    host_totals = {
        "api": 0.0,
        "command": 0.0,
        "driver": 0.0,
        "runtime": 0.0,
        "kernel": 0.0,
        "memcpy": 0.0,
        "graph": 0.0,
    }

    for stream_item, new_warnings in stream_chrome_events(input_path, chunk_size=chunk_size):
        if stream_item is None:
            for warning in new_warnings:
                if warning not in warnings:
                    warnings.append(warning)
            continue
        event = stream_item
        event_count += 1
        for warning in new_warnings:
            if warning not in warnings:
                warnings.append(warning)
        name = str(event.get("name") or "")
        duration = _number(event, ("dur", "duration"))
        start = _number(event, ("ts", "start", "start_us"))
        if duration is None or duration <= 0 or start is None:
            continue
        category = classify_event(event)
        pid = _number(event, ("pid",))
        tid = _number(event, ("tid",))
        args = event.get("args")
        args_map = args if isinstance(args, dict) else {}
        stream = _first_string(args_map, ("stream", "stream_id", "streamId"))
        correlation = _correlation(event)
        lowered = name.lower()

        scheduler_family = canonical_scheduler_family(name)
        if scheduler_family is not None:
            family = scheduler_family
            key = (family, name)
            aggregate = scheduler.setdefault(
                key,
                {"total_us": 0.0, "count": 0, "max_us": 0.0},
            )
            aggregate["total_us"] += duration
            aggregate["count"] += 1
            aggregate["max_us"] = max(float(aggregate["max_us"]), duration)
            rank = _number(event, ("rank",))
            if rank is not None:
                scheduler_ranks.setdefault(key, set()).add(int(rank))

        if GRAPH_RE.search(name):
            host_totals["graph"] += duration
        if lowered.find("launch") >= 0:
            host_totals["command"] += duration
        if API_RE.search(name):
            host_totals["api"] += duration
            host_totals["driver"] += duration
            if lowered.find("launch") < 0:
                host_totals["runtime"] += duration
        if KERNEL_RE.search(name):
            host_totals["kernel"] += duration
        if MEMCPY_RE.search(name):
            host_totals["memcpy"] += duration

        if category is None:
            continue
        classified += 1
        connection.execute(
            "INSERT INTO events(category, start_us, end_us, duration_us, pid, tid, stream, correlation_id, name, is_graph) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                category,
                float(start),
                float(start + duration),
                float(duration),
                int(pid) if pid is not None else None,
                int(tid) if tid is not None else None,
                stream,
                correlation,
                name,
                1 if GRAPH_RE.search(name) else 0,
            ),
        )

    connection.commit()
    trace_bounds = connection.execute("SELECT MIN(start_us), MAX(end_us) FROM events").fetchone()
    trace_start = float(trace_bounds[0]) if trace_bounds[0] is not None else None
    trace_end = float(trace_bounds[1]) if trace_bounds[1] is not None else None
    api_submit = _interval_total(connection, "api_submit")
    gpu_execution = _gpu_interval_total(connection)
    sync_wait = _interval_total(connection, "sync")
    graph_launch = _interval_total(connection, "graph_launch")
    graph_device = _graph_device_total(connection)
    explicit_pairs = _paired_latencies(connection, explicit_only=True)
    if explicit_pairs[2] > 0:
        launch_total, after_total, paired, unpaired = explicit_pairs
        evidence_mode = "EXCLUSIVE_UNION_WITH_EXPLICIT_CORRELATION"
        correlation_status = "EXPLICIT"
    else:
        adjacent = _paired_latencies(connection, explicit_only=False)
        launch_total, after_total, paired, unpaired = adjacent
        evidence_mode = "EXCLUSIVE_UNION_WITH_ADJACENT_GUESS"
        correlation_status = "UNKNOWN"
        if unpaired:
            warnings.append("no_explicit_correlation_ids_launch_latency_is_adjacent_guess")

    scheduler_signals = [
        _scheduler_signal(key, scheduler[key], scheduler_ranks.get(key, set()))
        for key in sorted(scheduler)
    ]
    if not scheduler_signals:
        warnings.append(
            "no_scheduler_annotations: host gaps are not scheduler evidence"
        )

    attribution = TimelineAttribution(
        schema_version=TIMELINE_SCHEMA,
        evidence_mode=evidence_mode,
        event_count=event_count,
        classified_event_count=classified,
        api_submit_us=api_submit,
        gpu_execution_us=gpu_execution,
        launch_latency_us=launch_total,
        sync_wait_us=sync_wait,
        graph_launch_us=graph_launch,
        graph_replay_device_us=graph_device,
        cpu_gap_before_us=launch_total,
        cpu_gap_after_us=after_total,
        paired_event_count=paired,
        unpaired_event_count=unpaired,
        correlation_status=correlation_status,
        host_totals_us=host_totals,
        scheduler_signals=scheduler_signals,
        warnings=warnings,
        database=str(database_path),
        trace_start_us=trace_start,
        trace_end_us=trace_end,
        graph_replay_boundaries=_graph_replay_boundaries(connection),
    )
    connection.close()
    return attribution
