"""Nsight Systems SQLite export adapter for NVIDIA TP captures.

The adapter emits hcTracer-compatible Chrome traces so the existing streaming
timeline engine can consume CUDA runtime/device events without changing the
profile contracts.  Nsight's ``globalPid``/``globalTid`` values contain the
Nsight process id in bits 24 and above; that stable id is used as the trace pid.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


TRUTH_SCHEMA = "infiniprofiler-nsys-truth-1"


class NsysExportError(RuntimeError):
    """The Nsight SQLite export cannot produce a matched TP capture."""


@dataclass
class NsysRankTruth:
    rank: int
    pid_key: int
    device_ids: List[int]
    gpu_execution_us: float
    runtime_api_us: float
    graph_launch_us: float
    sync_wait_us: float
    kernel_event_count: int
    memcpy_event_count: int
    memset_event_count: int
    runtime_event_count: int
    graph_launch_event_count: int
    sync_event_count: int
    paired_device_event_count: int
    unpaired_device_event_count: int
    correlation_status: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _connect_readonly(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise NsysExportError(f"Nsight SQLite export does not exist: {path}")
    uri = path.resolve().as_uri().replace("file:///", "file:/", 1)
    try:
        return sqlite3.connect(f"{uri}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise NsysExportError(f"cannot open Nsight SQLite export {path}: {exc}") from exc


def _table_exists(connection: sqlite3.Connection, name: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def _pid_key(global_id: int) -> int:
    return int(global_id) >> 24


def _rank_mapping(connection: sqlite3.Connection) -> List[Tuple[int, int, List[int]]]:
    rows = connection.execute(
        """
        WITH appearances AS (
          SELECT ((globalPid >> 24) & 0xffffffff) AS pid_key, deviceId AS device_id
            FROM CUPTI_ACTIVITY_KIND_KERNEL
          UNION ALL
          SELECT ((globalPid >> 24) & 0xffffffff), deviceId
            FROM CUPTI_ACTIVITY_KIND_MEMCPY
          UNION ALL
          SELECT ((globalPid >> 24) & 0xffffffff), deviceId
            FROM CUPTI_ACTIVITY_KIND_MEMSET
        )
        SELECT pid_key, MIN(device_id), GROUP_CONCAT(DISTINCT device_id)
          FROM appearances
         GROUP BY pid_key
         ORDER BY MIN(device_id), pid_key
        """
    ).fetchall()
    return [(int(pid), int(first_device), sorted({int(v) for v in devices.split(",")})) for pid, first_device, devices in rows]


def _require_tp(mapping: Sequence[Tuple[int, int, List[int]]], expected: int) -> None:
    if len(mapping) != expected:
        raise NsysExportError(f"expected {expected} CUDA worker processes, found {len(mapping)}")
    seen_devices: Dict[int, int] = {}
    for rank, (_, first_device, _) in enumerate(mapping):
        if first_device in seen_devices:
            raise NsysExportError(
                f"ranks {seen_devices[first_device]} and {rank} both use first CUDA device {first_device}"
            )
        seen_devices[first_device] = rank


def _write_rank_trace(
    connection: sqlite3.Connection,
    output_path: Path,
    pid_key: int,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    connection.execute("DROP TABLE IF EXISTS temp.rank_events")
    connection.execute(
        """
        CREATE TEMP TABLE rank_events (
            start_ns INTEGER, end_ns INTEGER, correlation_id INTEGER, event_name TEXT,
            event_pid INTEGER, event_tid INTEGER, stream_id INTEGER, device_id INTEGER,
            graph_node_id INTEGER, category TEXT
        )
        """
    )
    connection.execute(
        """
        INSERT INTO rank_events
        SELECT r.start, r.end, r.correlationId, COALESCE(s.value, 'cudaRuntimeAPI'),
               ((r.globalTid >> 24) & 0xffffffff), (r.globalTid & 0xffff), NULL, NULL,
               NULL, 'runtime'
          FROM CUPTI_ACTIVITY_KIND_RUNTIME r LEFT JOIN StringIds s ON s.id=r.nameId
        """
    )
    connection.execute(
        """
        INSERT INTO rank_events
        SELECT k.start, k.end, k.correlationId,
               CASE WHEN k.graphNodeId IS NULL THEN COALESCE(s.value, 'cudaKernel')
                    ELSE 'cudaGraphNode:' || COALESCE(s.value, 'cudaKernel') END,
               ((k.globalPid >> 24) & 0xffffffff), NULL, k.streamId, k.deviceId,
               k.graphNodeId, 'kernel'
          FROM CUPTI_ACTIVITY_KIND_KERNEL k LEFT JOIN StringIds s ON s.id=k.demangledName
        """
    )
    connection.execute(
        """
        INSERT INTO rank_events
        SELECT start, end, correlationId, 'cudaMemcpy', ((globalPid >> 24) & 0xffffffff),
               NULL, streamId, deviceId, graphNodeId, 'memcpy'
          FROM CUPTI_ACTIVITY_KIND_MEMCPY
        """
    )
    connection.execute(
        """
        INSERT INTO rank_events
        SELECT start, end, correlationId, 'cudaMemset', ((globalPid >> 24) & 0xffffffff),
               NULL, streamId, deviceId, graphNodeId, 'memset'
          FROM CUPTI_ACTIVITY_KIND_MEMSET
        """
    )
    connection.execute(
        """
        INSERT INTO rank_events
        SELECT start, end, correlationId, 'cudaSynchronize', ((globalPid >> 24) & 0xffffffff),
               NULL, streamId, deviceId, NULL, 'sync'
          FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION
        """
    )
    query = "SELECT * FROM temp.rank_events WHERE event_pid=? ORDER BY start_ns, category"
    cursor = connection.execute(query, (pid_key,))
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write('{"traceEvents": [')
        for start, end, correlation, name, event_pid, event_tid, stream, device, graph_node, category in cursor:
            duration_ns = max(0, int(end) - int(start))
            if duration_ns <= 0:
                continue
            event = {
                "ph": "X",
                "cat": category,
                "name": name,
                "ts": int(start) / 1000.0,
                "dur": duration_ns / 1000.0,
                "pid": int(event_pid),
                "tid": int(event_tid) if event_tid is not None else 0,
                "args": {
                    "correlation": int(correlation) if correlation is not None else None,
                    "stream": int(stream) if stream is not None else None,
                    "device": int(device) if device is not None else None,
                    "graph_node": int(graph_node) if graph_node is not None else None,
                },
            }
            handle.write(("," if count else "") + json.dumps(event, separators=(",", ":")))
            count += 1
        handle.write("]}\n")
    return count


def _union_ns(connection: sqlite3.Connection, rank: int) -> float:
    connection.execute("DROP TABLE IF EXISTS temp.rank_intervals")
    connection.execute(
        """
        CREATE TEMP TABLE rank_intervals (
            start_ns INTEGER NOT NULL,
            end_ns INTEGER NOT NULL
        )
        """
    )
    connection.execute(
        """
        INSERT INTO temp.rank_intervals(start_ns, end_ns)
          SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL
           WHERE ((globalPid >> 24) & 0xffffffff)=?
          UNION ALL SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMCPY
           WHERE ((globalPid >> 24) & 0xffffffff)=?
          UNION ALL SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMSET
           WHERE ((globalPid >> 24) & 0xffffffff)=?
        """,
        (rank, rank, rank),
    )
    active_start: Optional[int] = None
    active_end = 0
    total = 0
    for start, end in connection.execute(
        "SELECT start_ns, end_ns FROM temp.rank_intervals ORDER BY start_ns, end_ns"
    ):
        if end <= start:
            continue
        if active_start is None:
            active_start, active_end = start, end
        elif start <= active_end:
            active_end = max(active_end, end)
        else:
            total += active_end - active_start
            active_start, active_end = start, end
    if active_start is not None:
        total += active_end - active_start
    return total / 1000.0


def _runtime_union_ns(connection: sqlite3.Connection, pid_key: int) -> float:
    connection.execute("DROP TABLE IF EXISTS temp.runtime_intervals")
    connection.execute(
        "CREATE TEMP TABLE runtime_intervals (start_ns INTEGER NOT NULL, end_ns INTEGER NOT NULL)"
    )
    connection.execute(
        """
        INSERT INTO temp.runtime_intervals(start_ns, end_ns)
          SELECT r.start, r.end FROM CUPTI_ACTIVITY_KIND_RUNTIME r
           LEFT JOIN StringIds s ON s.id=r.nameId
           WHERE ((r.globalTid >> 24) & 0xffffffff)=?
             AND lower(COALESCE(s.value,'')) NOT LIKE '%synchron%'
             AND lower(COALESCE(s.value,'')) NOT LIKE '%eventrecord%'
             AND lower(COALESCE(s.value,'')) NOT LIKE '%streamwait%'
             AND lower(COALESCE(s.value,'')) NOT LIKE '%wait%'
             AND lower(COALESCE(s.value,'')) NOT LIKE '%barrier%'
        """,
        (pid_key,),
    )
    active_start: Optional[int] = None
    active_end = 0
    total = 0
    for start, end in connection.execute(
        "SELECT start_ns, end_ns FROM temp.runtime_intervals ORDER BY start_ns, end_ns"
    ):
        if end <= start:
            continue
        if active_start is None:
            active_start, active_end = start, end
        elif start <= active_end:
            active_end = max(active_end, end)
        else:
            total += active_end - active_start
            active_start, active_end = start, end
    if active_start is not None:
        total += active_end - active_start
    return total / 1000.0


def _sync_union_ns(connection: sqlite3.Connection, pid_key: int) -> float:
    connection.execute("DROP TABLE IF EXISTS temp.sync_intervals")
    connection.execute(
        "CREATE TEMP TABLE sync_intervals (start_ns INTEGER NOT NULL, end_ns INTEGER NOT NULL)"
    )
    connection.execute(
        """
        INSERT INTO temp.sync_intervals(start_ns, end_ns)
          SELECT start, end FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION
           WHERE ((globalPid >> 24) & 0xffffffff)=?
        """,
        (pid_key,),
    )
    connection.execute(
        """
        INSERT INTO temp.sync_intervals(start_ns, end_ns)
          SELECT r.start, r.end FROM CUPTI_ACTIVITY_KIND_RUNTIME r
           LEFT JOIN StringIds s ON s.id=r.nameId
           WHERE ((r.globalTid >> 24) & 0xffffffff)=?
             AND (lower(COALESCE(s.value,'')) LIKE '%synchron%'
                  OR lower(COALESCE(s.value,'')) LIKE '%eventrecord%'
                  OR lower(COALESCE(s.value,'')) LIKE '%streamwait%'
                  OR lower(COALESCE(s.value,'')) LIKE '%wait%'
                  OR lower(COALESCE(s.value,'')) LIKE '%barrier%')
        """,
        (pid_key,),
    )
    active_start: Optional[int] = None
    active_end = 0
    total = 0
    for start, end in connection.execute(
        "SELECT start_ns, end_ns FROM temp.sync_intervals ORDER BY start_ns, end_ns"
    ):
        if end <= start:
            continue
        if active_start is None:
            active_start, active_end = start, end
        elif start <= active_end:
            active_end = max(active_end, end)
        else:
            total += active_end - active_start
            active_start, active_end = start, end
    if active_start is not None:
        total += active_end - active_start
    return total / 1000.0


def _runtime_class_union_ns(
    connection: sqlite3.Connection,
    pid_key: int,
    *,
    graph: bool,
) -> float:
    table = "runtime_graph_intervals" if graph else "runtime_api_intervals"
    connection.execute(f"DROP TABLE IF EXISTS temp.{table}")
    connection.execute(
        f"CREATE TEMP TABLE {table} (start_ns INTEGER NOT NULL, end_ns INTEGER NOT NULL)"
    )
    if graph:
        predicate = "lower(COALESCE(s.value,'')) LIKE '%graph%'"
    else:
        predicate = """
            lower(COALESCE(s.value,'')) NOT LIKE '%synchron%'
            AND lower(COALESCE(s.value,'')) NOT LIKE '%eventrecord%'
            AND lower(COALESCE(s.value,'')) NOT LIKE '%streamwait%'
            AND lower(COALESCE(s.value,'')) NOT LIKE '%wait%'
            AND lower(COALESCE(s.value,'')) NOT LIKE '%barrier%'
            AND lower(COALESCE(s.value,'')) NOT LIKE '%graph%'
        """
    connection.execute(
        f"""
        INSERT INTO temp.{table}(start_ns, end_ns)
          SELECT r.start, r.end FROM CUPTI_ACTIVITY_KIND_RUNTIME r
           LEFT JOIN StringIds s ON s.id=r.nameId
           WHERE ((r.globalTid >> 24) & 0xffffffff)=? AND {predicate}
        """,
        (pid_key,),
    )
    active_start: Optional[int] = None
    active_end = 0
    total = 0
    for start, end in connection.execute(
        f"SELECT start_ns, end_ns FROM temp.{table} ORDER BY start_ns, end_ns"
    ):
        if end <= start:
            continue
        if active_start is None:
            active_start, active_end = start, end
        elif start <= active_end:
            active_end = max(active_end, end)
        else:
            total += active_end - active_start
            active_start, active_end = start, end
    if active_start is not None:
        total += active_end - active_start
    return total / 1000.0


def _device_pair_counts(connection: sqlite3.Connection, pid_key: int) -> Tuple[int, int]:
    connection.execute("DROP TABLE IF EXISTS temp.truth_device_events")
    connection.execute(
        "CREATE TEMP TABLE truth_device_events (correlation_id INTEGER)"
    )
    connection.execute(
        """
        INSERT INTO temp.truth_device_events(correlation_id)
          SELECT correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL
           WHERE ((globalPid >> 24) & 0xffffffff)=?
          UNION ALL SELECT correlationId FROM CUPTI_ACTIVITY_KIND_MEMCPY
           WHERE ((globalPid >> 24) & 0xffffffff)=?
          UNION ALL SELECT correlationId FROM CUPTI_ACTIVITY_KIND_MEMSET
           WHERE ((globalPid >> 24) & 0xffffffff)=?
        """,
        (pid_key, pid_key, pid_key),
    )
    paired, unpaired = connection.execute(
        """
        WITH runtime_correlations AS (
          SELECT DISTINCT correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME
           WHERE correlationId IS NOT NULL
             AND ((globalTid >> 24) & 0xffffffff)=?
        )
        SELECT
          SUM(CASE WHEN correlation_id IS NOT NULL AND correlation_id IN
                   (SELECT correlationId FROM runtime_correlations)
               THEN 1 ELSE 0 END),
          SUM(CASE WHEN correlation_id IS NULL OR correlation_id NOT IN
                   (SELECT correlationId FROM runtime_correlations)
               THEN 1 ELSE 0 END)
          FROM temp.truth_device_events
        """,
        (pid_key,),
    ).fetchone()
    return int(paired or 0), int(unpaired or 0)


def summarize_truth(path: Path, expected_ranks: int) -> Dict[str, Any]:
    """Return exclusive per-rank interval facts directly from Nsight SQLite."""
    connection = _connect_readonly(path)
    try:
        required = (
            "CUPTI_ACTIVITY_KIND_KERNEL",
            "CUPTI_ACTIVITY_KIND_RUNTIME",
            "CUPTI_ACTIVITY_KIND_MEMCPY",
            "CUPTI_ACTIVITY_KIND_MEMSET",
            "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION",
            "StringIds",
        )
        missing = [name for name in required if not _table_exists(connection, name)]
        if missing:
            raise NsysExportError(f"Nsight export is missing tables: {', '.join(missing)}")

        mapping = _rank_mapping(connection)
        _require_tp(mapping, expected_ranks)
        rank_facts: List[Dict[str, Any]] = []
        for rank, (pid_key, first_device, devices) in enumerate(mapping):
            gpu_us = _union_ns(connection, pid_key)

            def scalar(query: str) -> Tuple[float, int]:
                value, count = connection.execute(query, (pid_key,)).fetchone()
                return (float(value) / 1000.0, int(count))

            runtime_us = _runtime_union_ns(connection, pid_key)
            runtime_count = connection.execute(
                "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE ((globalTid >> 24) & 0xffffffff)=?",
                (pid_key,),
            ).fetchone()[0]
            runtime_api_us = _runtime_class_union_ns(connection, pid_key, graph=False)
            graph_launch_us = _runtime_class_union_ns(connection, pid_key, graph=True)
            graph_launch_count = connection.execute(
                """
                SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME r
                 LEFT JOIN StringIds s ON s.id=r.nameId
                 WHERE ((r.globalTid >> 24) & 0xffffffff)=?
                   AND lower(COALESCE(s.value,'')) LIKE '%graph%'
                """,
                (pid_key,),
            ).fetchone()[0]
            sync_us = _sync_union_ns(connection, pid_key)
            sync_count = connection.execute(
                "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION WHERE ((globalPid >> 24) & 0xffffffff)=?",
                (pid_key,),
            ).fetchone()[0]
            kernel_count = connection.execute(
                "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE ((globalPid >> 24) & 0xffffffff)=?",
                (pid_key,),
            ).fetchone()[0]
            memcpy_count = connection.execute(
                "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE ((globalPid >> 24) & 0xffffffff)=?",
                (pid_key,),
            ).fetchone()[0]
            memset_count = connection.execute(
                "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_MEMSET WHERE ((globalPid >> 24) & 0xffffffff)=?",
                (pid_key,),
            ).fetchone()[0]
            paired_count, unpaired_count = _device_pair_counts(connection, pid_key)
            correlation_status = "EXPLICIT" if unpaired_count == 0 else "PARTIAL"
            rank_facts.append(
                NsysRankTruth(
                    rank=rank,
                    pid_key=pid_key,
                    device_ids=devices,
                    gpu_execution_us=gpu_us,
                    runtime_api_us=runtime_api_us,
                    graph_launch_us=graph_launch_us,
                    sync_wait_us=sync_us,
                    kernel_event_count=int(kernel_count),
                    memcpy_event_count=int(memcpy_count),
                    memset_event_count=int(memset_count),
                    runtime_event_count=int(runtime_count),
                    graph_launch_event_count=int(graph_launch_count),
                    sync_event_count=int(sync_count),
                    paired_device_event_count=paired_count,
                    unpaired_device_event_count=unpaired_count,
                    correlation_status=correlation_status,
                ).to_dict()
            )
        return {
            "schema_version": TRUTH_SCHEMA,
            "source": str(path),
            "expected_rank_count": expected_ranks,
            "rank_count": len(rank_facts),
            "ranks": rank_facts,
        }
    finally:
        connection.close()


def export_rank_traces(
    sqlite_path: Path,
    output_dir: Path,
    *,
    expected_ranks: int,
) -> Dict[str, Any]:
    """Split one Nsight SQLite export into per-rank Chrome traces."""
    truth = summarize_truth(sqlite_path, expected_ranks)
    connection = _connect_readonly(sqlite_path)
    try:
        rank_files: List[Dict[str, Any]] = []
        for fact in truth["ranks"]:
            rank = int(fact["rank"])
            trace_path = output_dir / f"rank={rank:02d}" / "raw" / "tracer_out-0.json"
            event_count = _write_rank_trace(connection, trace_path, int(fact["pid_key"]))
            rank_files.append(
                {
                    "rank": rank,
                    "pid_key": int(fact["pid_key"]),
                    "trace_path": str(trace_path),
                    "event_count": event_count,
                    "bytes": trace_path.stat().st_size,
                }
            )
    finally:
        connection.close()
    return {"truth": truth, "rank_files": rank_files}


def timing_attribution_from_rank_truth(rank: Mapping[str, Any]) -> Dict[str, Any]:
    """Map one NsysRankTruth row into the shared timing_attribution object."""
    unpaired = int(rank.get("unpaired_device_event_count") or 0)
    paired = int(rank.get("paired_device_event_count") or 0)
    correlation = str(rank.get("correlation_status") or "UNKNOWN")
    evidence_mode = (
        "EXCLUSIVE_UNION_WITH_EXPLICIT_CORRELATION"
        if correlation == "EXPLICIT" and unpaired == 0
        else "EXCLUSIVE_UNION_WITH_ADJACENT_GUESS"
    )
    runtime_api = float(rank.get("runtime_api_us") or 0.0)
    return {
        "evidence_mode": evidence_mode,
        "api_submit_us": runtime_api,
        "gpu_execution_us": float(rank.get("gpu_execution_us") or 0.0),
        "launch_latency_us": 0.0,
        "sync_wait_us": float(rank.get("sync_wait_us") or 0.0),
        "graph_launch_us": float(rank.get("graph_launch_us") or 0.0),
        "graph_replay_device_us": 0.0,
        "cpu_gap_before_us": 0.0,
        "cpu_gap_after_us": 0.0,
        "unpaired_event_count": unpaired,
        "paired_event_count": paired,
    }


def timing_attribution_from_truth(truth: Mapping[str, Any]) -> Dict[str, Any]:
    """Aggregate per-rank Nsight truth into one timing_attribution object."""
    ranks = truth.get("ranks") if isinstance(truth.get("ranks"), list) else []
    if not ranks:
        return timing_attribution_from_rank_truth({})
    totals = {
        "api_submit_us": 0.0,
        "gpu_execution_us": 0.0,
        "sync_wait_us": 0.0,
        "graph_launch_us": 0.0,
        "unpaired_event_count": 0,
        "paired_event_count": 0,
    }
    explicit = True
    for rank in ranks:
        if not isinstance(rank, Mapping):
            continue
        item = timing_attribution_from_rank_truth(rank)
        totals["api_submit_us"] += float(item["api_submit_us"])
        totals["gpu_execution_us"] += float(item["gpu_execution_us"])
        totals["sync_wait_us"] += float(item["sync_wait_us"])
        totals["graph_launch_us"] += float(item["graph_launch_us"])
        totals["unpaired_event_count"] += int(item["unpaired_event_count"])
        totals["paired_event_count"] += int(item["paired_event_count"])
        if item["evidence_mode"] != "EXCLUSIVE_UNION_WITH_EXPLICIT_CORRELATION":
            explicit = False
    return {
        "evidence_mode": (
            "EXCLUSIVE_UNION_WITH_EXPLICIT_CORRELATION"
            if explicit
            else "EXCLUSIVE_UNION_WITH_ADJACENT_GUESS"
        ),
        "api_submit_us": totals["api_submit_us"],
        "gpu_execution_us": totals["gpu_execution_us"],
        "launch_latency_us": 0.0,
        "sync_wait_us": totals["sync_wait_us"],
        "graph_launch_us": totals["graph_launch_us"],
        "graph_replay_device_us": 0.0,
        "cpu_gap_before_us": 0.0,
        "cpu_gap_after_us": 0.0,
        "unpaired_event_count": totals["unpaired_event_count"],
        "paired_event_count": totals["paired_event_count"],
    }


def normalize(
    input_path: Path,
    *,
    expected_ranks: int = 1,
    mode: str = "nsys",
    baseline_path: Optional[Path] = None,
) -> "ProfileDocument":
    """Normalize an Nsight SQLite export into a timing-first profile document."""
    from ..models import Capture, HostSignal, ProfileDocument, TimingAttributionFields, safe_ratio

    truth = summarize_truth(input_path, expected_ranks)
    timing_raw = timing_attribution_from_truth(truth)
    baseline_timing = None
    warnings: List[str] = []
    if baseline_path is not None:
        baseline_truth = summarize_truth(baseline_path, expected_ranks)
        baseline_timing = timing_attribution_from_truth(baseline_truth)

    host_signals: List[HostSignal] = []
    for family, key in (
        ("api", "api_submit_us"),
        ("runtime", "api_submit_us"),
        ("graph", "graph_launch_us"),
    ):
        candidate = float(timing_raw.get(key) or 0.0)
        baseline = float(baseline_timing.get(key) or 0.0) if baseline_timing else None
        if candidate <= 0 and (baseline is None or baseline <= 0):
            continue
        host_signals.append(
            HostSignal(
                family=family,
                candidate=candidate,
                baseline=baseline,
                ratio=safe_ratio(candidate, baseline) if baseline else None,
                source="nsys",
            )
        )

    provenance: Dict[str, Any] = {
        "input": str(input_path),
        "expected_ranks": expected_ranks,
        "rank_count": truth.get("rank_count"),
        "nsys_truth_schema": truth.get("schema_version"),
    }
    if baseline_timing is not None:
        provenance["baseline_timing_attribution"] = baseline_timing
        provenance["baseline_input"] = str(baseline_path)

    return ProfileDocument(
        capture=Capture(source="nsys", mode=mode, provenance=provenance),
        host_signals=host_signals,
        timing_attribution=TimingAttributionFields.from_mapping(timing_raw),
        warnings=warnings,
    )
