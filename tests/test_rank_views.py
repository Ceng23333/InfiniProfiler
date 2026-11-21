import json
from pathlib import Path

import pytest

from infiniprofiler.rank_views import (
    RankViewError,
    build_rank_views,
    build_rank_views_from_capture,
)


def timing(
    *,
    trace_start,
    trace_end,
    busy,
    boundaries,
    correlation="EXPLICIT",
):
    return {
        "gpu_execution_us": busy,
        "trace_start_us": trace_start,
        "trace_end_us": trace_end,
        "correlation_status": correlation,
        "graph_replay_boundaries": boundaries,
    }


def make_timing(trace_start, trace_end, busy, boundaries):
    return timing(
        trace_start=trace_start,
        trace_end=trace_end,
        busy=busy,
        boundaries=boundaries,
    )


def manifest_and_merge():
    manifest = {
        "schema_version": "infiniprofiler-capture-manifest-1",
        "status": "SUCCEEDED",
        "started_epoch_ns": 1_000_000,
        "ended_epoch_ns": 2_000_000,
        "duration_ns": 1_000_000,
        "spec": {
            "campaign_id": "qwen3-8b",
            "tensor_parallel": 2,
        },
        "ranks": [
            {"rank": 0, "pid": 1000, "status": "SUCCEEDED"},
            {"rank": 1, "pid": 1001, "status": "SUCCEEDED"},
        ],
    }
    boundary_rank_0 = {
        "pid": 1000,
        "correlation_id": "g1",
        "launch_count": 1,
        "device_event_count": 2,
        "start_us": 15,
        "end_us": 120,
        "gpu_busy_us": 120,
    }
    boundary_rank_1 = {
        "pid": 1001,
        "correlation_id": "g1",
        "launch_count": 1,
        "device_event_count": 2,
        "start_us": 10,
        "end_us": 100,
        "gpu_busy_us": 100,
    }
    merge = {
        "schema_version": "infiniprofiler-capture-merge-1",
        "status": "COMPLETE",
        "manifest": "/tmp/window_manifest.json",
        "window_id": "window-0",
        "role": "candidate",
        "ranks": [
            {
                "rank": 0,
                "pid": 1000,
                "timing_attribution": make_timing(100, 200, 120, [boundary_rank_0]),
            },
            {
                "rank": 1,
                "pid": 1001,
                "timing_attribution": make_timing(110, 190, 100, [boundary_rank_1]),
            },
        ],
        "warnings": [],
    }
    return manifest, merge


def test_matched_graph_boundaries_report_straggler_and_imbalance(tmp_path):
    manifest, merge = manifest_and_merge()
    result = build_rank_views(merge, manifest)

    assert result.status == "MATCHED_GRAPH_REPLAY"
    assert result.boundary_status == "MATCHED_GRAPH_REPLAY"
    assert result.correlation_status == "EXPLICIT"
    assert [rank.rank for rank in result.rank_views] == [0, 1]
    assert result.rank_views[0].start_skew_us == 5
    assert result.rank_views[1].start_skew_us == 0
    assert result.rank_views[0].last_finish_skew_us == 20
    assert result.rank_views[1].last_finish_skew_us == 0
    assert result.rank_views[1].straggler_wait_us == 20
    assert result.rank_views[0].straggler_wait_us == 0
    assert result.rank_views[0].tp_imbalance_ratio == pytest.approx(120 / 110)
    assert result.rank_views[0].granularity == "graph_replay_boundary"
    assert result.rank_views[0].boundary_status == "MATCHED_GRAPH_REPLAY"


def test_unmatched_boundaries_suppress_straggler_claims(tmp_path):
    manifest, merge = manifest_and_merge()
    merge["ranks"][1]["timing_attribution"]["graph_replay_boundaries"] = []
    merge["ranks"][1]["timing_attribution"]["correlation_status"] = "UNKNOWN"

    result = build_rank_views(merge, manifest)

    assert result.status == "COARSE_ONLY"
    assert result.boundary_status == "UNMATCHED"
    assert result.correlation_status == "UNKNOWN"
    assert all(rank.straggler_wait_us == 0 for rank in result.rank_views)
    assert "graph_replay_boundaries_not_matched_straggler_wait_suppressed" in result.warnings


def test_rejects_partial_expected_ranks(tmp_path):
    manifest, merge = manifest_and_merge()
    merge["ranks"] = merge["ranks"][:1]

    with pytest.raises(RankViewError, match="exactly 2 ranks"):
        build_rank_views(merge, manifest)


def test_capture_directory_writes_rank_views(tmp_path):
    manifest, merge = manifest_and_merge()
    (tmp_path / "window_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "capture_merge.json").write_text(json.dumps(merge), encoding="utf-8")
    output = tmp_path / "rank_views.json"

    result = build_rank_views_from_capture(tmp_path, output)

    assert output.is_file()
    assert result.status == "MATCHED_GRAPH_REPLAY"
