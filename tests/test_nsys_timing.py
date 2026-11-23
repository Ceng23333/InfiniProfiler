"""Nsight rank truth maps onto the shared timing_attribution buckets."""

from infiniprofiler.adapters.nsys_sqlite import (
    timing_attribution_from_rank_truth,
    timing_attribution_from_truth,
)


def test_rank_truth_maps_nsys_families():
    timing = timing_attribution_from_rank_truth(
        {
            "runtime_api_us": 12.5,
            "gpu_execution_us": 1000.0,
            "graph_launch_us": 8.0,
            "sync_wait_us": 4.0,
            "paired_device_event_count": 20,
            "unpaired_device_event_count": 0,
            "correlation_status": "EXPLICIT",
        }
    )

    assert timing["api_submit_us"] == 12.5
    assert timing["gpu_execution_us"] == 1000.0
    assert timing["graph_launch_us"] == 8.0
    assert timing["sync_wait_us"] == 4.0
    assert timing["paired_event_count"] == 20
    assert timing["evidence_mode"] == "EXCLUSIVE_UNION_WITH_EXPLICIT_CORRELATION"
    assert "scheduler_signals" not in timing


def test_truth_aggregate_sums_ranks_and_keeps_gate_families():
    timing = timing_attribution_from_truth(
        {
            "ranks": [
                {
                    "runtime_api_us": 10,
                    "gpu_execution_us": 100,
                    "graph_launch_us": 1,
                    "sync_wait_us": 2,
                    "paired_device_event_count": 5,
                    "unpaired_device_event_count": 0,
                    "correlation_status": "EXPLICIT",
                },
                {
                    "runtime_api_us": 30,
                    "gpu_execution_us": 300,
                    "graph_launch_us": 3,
                    "sync_wait_us": 6,
                    "paired_device_event_count": 7,
                    "unpaired_device_event_count": 1,
                    "correlation_status": "PARTIAL",
                },
            ]
        }
    )

    assert timing["api_submit_us"] == 40
    assert timing["gpu_execution_us"] == 400
    assert timing["graph_launch_us"] == 4
    assert timing["sync_wait_us"] == 8
    assert timing["paired_event_count"] == 12
    assert timing["unpaired_event_count"] == 1
    assert timing["evidence_mode"] == "EXCLUSIVE_UNION_WITH_ADJACENT_GUESS"
