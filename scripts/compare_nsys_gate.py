#!/usr/bin/env python3
"""Compare one Nsight truth run with one InfiniProfiler-normalized run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

from infiniprofiler.capture import compare_capture_manifests


METRIC_THRESHOLDS = {
    "gpu_execution_us": {"global": 0.03, "rank": 0.05},
    "runtime_api_us": {"global": 0.10, "rank": 0.15},
    "graph_launch_us": {"global": 0.10, "rank": 0.15},
    "sync_wait_us": {"global": 0.10, "rank": 0.15},
    "event_count": {"global": 0.02, "rank": 0.05},
    "paired_event_count": {"global": 0.02, "rank": 0.03},
}


def relative_delta(baseline: float, candidate: float) -> float:
    if baseline == 0:
        return 0.0 if candidate == 0 else math.inf
    return candidate / baseline - 1.0


def truth_event_count(row: Dict[str, Any]) -> int:
    return int(
        row["kernel_event_count"]
        + row["memcpy_event_count"]
        + row["memset_event_count"]
        + row["runtime_event_count"]
        + row["sync_event_count"]
    )


def timeline_row(merge: Dict[str, Any], rank: int) -> Dict[str, Any]:
    matches = [row for row in merge["ranks"] if int(row["rank"]) == rank]
    if not matches:
        raise ValueError(f"capture merge has no rank {rank}")
    return matches[0]


def metric_pair(truth: Dict[str, Any], merge: Dict[str, Any], rank: int, metric: str):
    truth_rank = next(row for row in truth["ranks"] if int(row["rank"]) == rank)
    merge_timing = timeline_row(merge, rank)["timing_attribution"]
    if metric == "event_count":
        baseline = float(truth_event_count(truth_rank))
        candidate = float(merge_timing["event_count"])
    elif metric == "gpu_execution_us":
        baseline = float(truth_rank[metric])
        candidate = float(merge_timing[metric])
    elif metric == "paired_event_count":
        baseline = float(truth_rank["paired_device_event_count"])
        candidate = float(merge_timing["paired_event_count"])
    elif metric == "runtime_api_us":
        baseline = float(truth_rank[metric])
        candidate = float(merge_timing["api_submit_us"])
    elif metric == "graph_launch_us":
        baseline = float(truth_rank[metric])
        candidate = float(merge_timing["graph_launch_us"])
    elif metric == "sync_wait_us":
        baseline = float(truth_rank[metric])
        candidate = float(merge_timing["sync_wait_us"])
    else:
        candidate = float(merge_timing[metric])
    return baseline, candidate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True, help="run directory containing profile/nsys_truth.json")
    parser.add_argument("--candidate", type=Path, required=True, help="run directory containing profile/capture_merge.json")
    parser.add_argument("--expected-ranks", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline_truth = json.loads((args.baseline / "profile/nsys_truth.json").read_text(encoding="utf-8"))
    candidate_truth = json.loads((args.candidate / "profile/nsys_truth.json").read_text(encoding="utf-8"))
    candidate_merge = json.loads((args.candidate / "profile/capture_merge.json").read_text(encoding="utf-8"))
    baseline_client = json.loads((args.baseline / "client/measured.json").read_text(encoding="utf-8"))
    candidate_client = json.loads((args.candidate / "client/measured.json").read_text(encoding="utf-8"))
    baseline_manifest_path = args.baseline / "profile/window_manifest.json"
    candidate_manifest_path = args.candidate / "profile/window_manifest.json"
    capture_manifest = compare_capture_manifests(baseline_manifest_path, candidate_manifest_path)

    errors: List[str] = []
    baseline_ranks = {int(row["rank"]): row for row in baseline_truth["ranks"]}
    candidate_truth_ranks = {int(row["rank"]): row for row in candidate_truth["ranks"]}
    if len(baseline_ranks) != args.expected_ranks or len(candidate_truth_ranks) != args.expected_ranks:
        errors.append(
            f"rank count baseline={len(baseline_ranks)} candidate={len(candidate_truth_ranks)} expected={args.expected_ranks}"
        )
    if set(baseline_ranks) != set(candidate_truth_ranks):
        errors.append(f"rank ids differ: baseline={sorted(baseline_ranks)} candidate={sorted(candidate_truth_ranks)}")
    if candidate_merge.get("status") != "COMPLETE":
        errors.append(f"candidate capture merge status={candidate_merge.get('status')}")
    if candidate_merge.get("rank_views_status") not in ("MATCHED_GRAPH_REPLAY", "COARSE_ONLY"):
        errors.append(f"candidate rank views status={candidate_merge.get('rank_views_status')}")
    if capture_manifest.get("status") != "MATCHED":
        errors.extend(f"capture manifest: {item}" for item in capture_manifest.get("errors", []))

    comparisons: List[Dict[str, Any]] = []
    for metric, thresholds in METRIC_THRESHOLDS.items():
        baseline_total = 0.0
        candidate_total = 0.0
        rank_rows: List[Dict[str, Any]] = []
        for rank in sorted(set(baseline_ranks) & set(candidate_truth_ranks)):
            baseline, candidate = metric_pair(baseline_truth, candidate_merge, rank, metric)
            baseline_total += baseline
            candidate_total += candidate
            delta = relative_delta(baseline, candidate)
            passed = abs(delta) <= thresholds["rank"]
            rank_rows.append(
                {
                    "rank": rank,
                    "baseline": baseline,
                    "candidate": candidate,
                    "relative_delta": delta,
                    "threshold": thresholds["rank"],
                    "status": "PASS" if passed else "FAIL",
                }
            )
            if not passed:
                errors.append(
                    f"rank={rank} {metric} relative_delta={delta:.6f} exceeds {thresholds['rank']:.6f}"
                )
        global_delta = relative_delta(baseline_total, candidate_total)
        global_passed = abs(global_delta) <= thresholds["global"]
        if not global_passed:
            errors.append(
                f"global {metric} relative_delta={global_delta:.6f} exceeds {thresholds['global']:.6f}"
            )
        comparisons.append(
            {
                "metric": metric,
                "baseline_total": baseline_total,
                "candidate_total": candidate_total,
                "relative_delta": global_delta,
                "threshold": thresholds["global"],
                "status": "PASS" if global_passed else "FAIL",
                "ranks": rank_rows,
            }
        )

    explicit_ranks = [
        timeline_row(candidate_merge, rank)["timing_attribution"]["correlation_status"]
        for rank in sorted(candidate_truth_ranks)
    ]
    if any(status != "EXPLICIT" for status in explicit_ranks):
        errors.append(f"candidate correlation statuses are not all EXPLICIT: {explicit_ranks}")

    wall = {
        "baseline": {
            "completed_requests": baseline_client["completed_requests"],
            "failed_requests": baseline_client["failed_requests"],
            "wall_duration_ms": baseline_client["wall_duration_ms"],
            "tokens_per_second": baseline_client["tokens_per_second"],
        },
        "candidate": {
            "completed_requests": candidate_client["completed_requests"],
            "failed_requests": candidate_client["failed_requests"],
            "wall_duration_ms": candidate_client["wall_duration_ms"],
            "tokens_per_second": candidate_client["tokens_per_second"],
        },
        "duration_ratio": candidate_client["wall_duration_ms"] / baseline_client["wall_duration_ms"],
        "tokens_per_second_ratio": candidate_client["tokens_per_second"] / baseline_client["tokens_per_second"],
    }
    if baseline_client["failed_requests"] or candidate_client["failed_requests"]:
        errors.append("one or both measured client windows had failed requests")
    if not 0.95 <= wall["duration_ratio"] <= 1.05:
        errors.append(f"client duration ratio {wall['duration_ratio']:.6f} outside 0.95-1.05")

    result = {
        "schema_version": "infiniprofiler-nsys-gate-comparison-1",
        "status": "PASS" if not errors else "FAIL",
        "expected_rank_count": args.expected_ranks,
        "capture_manifest_comparison": capture_manifest,
        "candidate_correlation_statuses": explicit_ranks,
        "client_wall": wall,
        "metric_comparisons": comparisons,
        "errors": errors,
        "inputs": {
            "baseline": str(args.baseline),
            "candidate": str(args.candidate),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "errors": errors}, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
