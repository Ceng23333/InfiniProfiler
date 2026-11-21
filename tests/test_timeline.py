import json
from pathlib import Path

import pytest

from infiniprofiler.timeline import (
    attribute_trace,
    stream_chrome_events,
)


def trace_file(tmp_path: Path, name: str, events, complete: bool = True) -> Path:
    path = tmp_path / name
    prefix = json.dumps({"traceEvents": events}, separators=(",", ":"))
    path.write_text(prefix if complete else prefix[:-2], encoding="utf-8")
    return path


def event(name, category, start, duration, *, stream="s0", correlation=None, pid=1):
    args = {"stream": stream, "cat": category}
    if correlation is not None:
        args["correlation_id"] = correlation
    return {
        "name": name,
        "ph": "X",
        "cat": category,
        "ts": start,
        "dur": duration,
        "pid": pid,
        "tid": 10,
        "args": args,
    }


def test_streaming_reader_recovers_truncated_tail(tmp_path):
    path = trace_file(
        tmp_path,
        "truncated.json",
        [
            {"name": "a", "ph": "X", "ts": 0, "dur": 1},
            {"name": "b", "ph": "X", "ts": 2, "dur": 3},
        ],
        complete=False,
    )
    events = []
    warnings = []
    for item, new_warnings in stream_chrome_events(path, chunk_size=64):
        if item is not None:
            events.append(item)
        warnings.extend(new_warnings)

    assert [event["name"] for event in events] == ["a", "b"]
    assert "recovered_truncated_chrome_trace" in warnings


def test_streaming_reader_reports_invalid_tail_event_before_array_close(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text(
        '{"traceEvents":[{"name":"a","ph":"X","ts":0,"dur":1},{broken},{"name":"b","ph":"X","ts":2,"dur":3}]}',
        encoding="utf-8",
    )
    warnings = []
    events = 0
    for item, new_warnings in stream_chrome_events(path, chunk_size=64):
        if item is not None:
            events += 1
        warnings.extend(new_warnings)

    assert events == 2
    assert "invalid_trace_event_skipped" in warnings


def test_same_stream_overlap_is_unioned(tmp_path):
    path = trace_file(
        tmp_path,
        "same.json",
        [
            event("kernel_a", "kernel", 0, 100),
            event("kernel_b", "kernel", 75, 50),
        ],
    )
    timeline = attribute_trace(path, database_path=tmp_path / "timeline.db")

    assert timeline.gpu_execution_us == 125
    assert timeline.classified_event_count == 2


def test_different_stream_overlap_is_rank_union_not_sum(tmp_path):
    path = trace_file(
        tmp_path,
        "streams.json",
        [
            event("kernel_a", "kernel", 0, 100, stream="s0"),
            event("kernel_b", "kernel", 50, 50, stream="s1"),
        ],
    )
    timeline = attribute_trace(path, database_path=tmp_path / "timeline.db")

    assert timeline.gpu_execution_us == 100


def test_graph_parent_and_child_do_not_double_count_device_time(tmp_path):
    path = trace_file(
        tmp_path,
        "graph.json",
        [
            event("cudaGraphLaunch", "runtime", 0, 20, correlation="g1"),
            event("graph_kernel_a", "kernel", 10, 30, stream="g", correlation="g1"),
            event("graph_kernel_b", "kernel", 25, 15, stream="g", correlation="g1"),
        ],
    )
    timeline = attribute_trace(path, database_path=tmp_path / "timeline.db")

    assert timeline.graph_launch_us == 20
    assert timeline.graph_replay_device_us == 30
    assert timeline.gpu_execution_us == 30
    assert timeline.evidence_mode == "EXCLUSIVE_UNION_WITH_EXPLICIT_CORRELATION"
    assert timeline.unpaired_event_count == 0
    assert timeline.launch_latency_us == 5


def test_explicit_correlation_pairs_submit_and_device(tmp_path):
    path = trace_file(
        tmp_path,
        "paired.json",
        [
            event("cuLaunchKernel", "runtime", 0, 10, correlation="c1"),
            event("attention_kernel", "kernel", 30, 50, correlation="c1"),
        ],
    )
    timeline = attribute_trace(path, database_path=tmp_path / "timeline.db")

    assert timeline.evidence_mode == "EXCLUSIVE_UNION_WITH_EXPLICIT_CORRELATION"
    assert timeline.correlation_status == "EXPLICIT"
    assert timeline.launch_latency_us == 20
    assert timeline.cpu_gap_before_us == 20
    assert timeline.paired_event_count == 1
    assert timeline.unpaired_event_count == 0


def test_unpaired_adjacency_is_marked_as_guess(tmp_path):
    path = trace_file(
        tmp_path,
        "adjacent.json",
        [
            event("cuLaunchKernel", "runtime", 0, 10),
            event("attention_kernel", "kernel", 20, 50),
            event("unrelated_kernel", "kernel", 2_000, 10),
        ],
    )
    timeline = attribute_trace(path, database_path=tmp_path / "timeline.db")

    assert timeline.evidence_mode == "EXCLUSIVE_UNION_WITH_ADJACENT_GUESS"
    assert timeline.correlation_status == "UNKNOWN"
    assert timeline.launch_latency_us == 10
    assert timeline.unpaired_event_count == 1
    assert "no_explicit_correlation_ids_launch_latency_is_adjacent_guess" in timeline.warnings


def test_scheduler_markers_are_collected_during_streaming(tmp_path):
    path = trace_file(
        tmp_path,
        "scheduler.json",
        [
            {
                "name": "infinilm:schedule",
                "ph": "X",
                "ts": 0,
                "dur": 40,
                "args": {"rank": 2},
            }
        ],
    )
    timeline = attribute_trace(path, database_path=tmp_path / "timeline.db")

    assert timeline.scheduler_signals[0]["family"] == "step_control"
    assert timeline.scheduler_signals[0]["candidate_us"] == 40
    assert timeline.scheduler_signals[0]["ranks"] == [2]


def test_streaming_reader_processes_large_trace_without_full_list(tmp_path):
    path = tmp_path / "large.json"
    with path.open("w", encoding="utf-8") as handle:
        handle.write('{"traceEvents":[')
        for index in range(20_000):
            if index:
                handle.write(",")
            handle.write(json.dumps({"name": f"kernel_{index}", "ph": "X", "ts": index, "dur": 1}))
        handle.write("]}")

    count = 0
    for item, _ in stream_chrome_events(path, chunk_size=4096):
        if item is not None:
            count += 1
    assert count == 20_000
