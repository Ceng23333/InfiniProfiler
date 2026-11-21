import json

import pytest

from infiniprofiler.contracts import (
    ContractValidationError,
    validate_profile_v2,
    validate_promotion_request_v2,
)


def valid_profile():
    return {
        "schema_version": "infiniprofiler-profile-2",
        "capture": {
            "source": "hctracer",
            "mode": "attach",
            "provenance": {"event_count": 12},
        },
        "window": {
            "window_id": "window-0",
            "wall_compare_ref": "suite/compare.json",
            "wall_workload_id": "qwen3-8b-decode-256",
            "client_started_epoch_ns": 1_730_000_000_000_000_000,
            "client_ended_epoch_ns": 1_730_000_001_000_000_000,
            "warmup_excluded": True,
            "start_skew_ms": 10,
            "duration_ratio": 1.0,
            "alignment": "MATCHED",
        },
        "workload": {
            "campaign_id": "qwen3-8b-metax9-sg",
            "model": "Qwen3-8B",
            "backend_baseline_pin": "vllm-frozen",
            "backend_candidate_pin": "infinilm-candidate",
            "tensor_parallel": 4,
            "concurrency": 8,
            "prompt_lengths": [512],
            "output_lengths": [256],
            "batch_size": 8,
            "clock_policy": "operator_observed",
            "profiles_source_hash": "sha256:candidate",
        },
        "ranks": [
            {
                "rank": 0,
                "pid": 1000,
                "gpu_busy_us": 900_000,
                "start_epoch_ns": 1_730_000_000_010_000_000,
                "end_epoch_ns": 1_730_000_001_000_000_000,
                "start_skew_us": 10,
                "last_finish_skew_us": 0,
                "straggler_wait_us": 0,
                "tp_imbalance_ratio": 1.0,
                "correlation_status": "EXPLICIT",
            },
            {
                "rank": 1,
                "pid": 1001,
                "gpu_busy_us": 950_000,
                "start_epoch_ns": 1_730_000_000_020_000_000,
                "end_epoch_ns": 1_730_000_001_010_000_000,
                "start_skew_us": 20,
                "last_finish_skew_us": 10_000,
                "straggler_wait_us": 10_000,
                "tp_imbalance_ratio": 1.05,
                "correlation_status": "UNKNOWN",
            },
        ],
        "timing_attribution": {
            "evidence_mode": "exclusive_union_with_paired_correlation",
            "api_submit_us": 20_000,
            "gpu_execution_us": 1_000_000,
            "launch_latency_us": 5_000,
            "sync_wait_us": 2_000,
            "graph_launch_us": 8_000,
            "graph_replay_device_us": 700_000,
            "cpu_gap_before_us": 1_000,
            "cpu_gap_after_us": 500,
            "unpaired_event_count": 1,
            "paired_event_count": 20,
        },
        "source_map": {
            "source_pin": "infinilm-candidate",
            "source_root_hash": "sha256:candidate",
            "symbols": [
                {
                    "symbol": "MhaKVCache::supports_device_graph_capture",
                    "file": "infinilm/attention/mha_kv_cache.cpp",
                    "kind": "graph_guard",
                }
            ],
            "providers": [
                {"provider": "flash_attn", "patterns": ["flash_attn"], "files": ["infinilm/attention"]}
            ],
        },
        "host_signals": [
            {
                "family": "graph",
                "candidate": 31.0,
                "baseline": 1.0,
                "ratio": 31.0,
                "source": "hctracer",
            }
        ],
        "scheduler_signals": [],
        "metrics": [],
        "warnings": [],
    }


def valid_request():
    return {
        "schema_version": "promotion-request-2",
        "controller": "autohelix",
        "stop_before_promote": True,
        "campaign": {
            "campaign_id": "qwen3-8b-metax9-sg",
            "model": "Qwen3-8B",
            "host": "metax-9-sg",
            "device_track": "mars",
            "baseline_pin": "vllm-frozen",
            "candidate_pin": "infinilm-candidate",
            "iteration": "0",
        },
        "observation": {
            "profile_ref": "profile/window-0/profile.json",
            "window": {"window_id": "window-0", "alignment": "MATCHED"},
        },
        "gate": {
            "metric": "unprofiled_primary_wall",
            "surpass_threshold": 1.05,
            "profiled_result_can_promote": False,
        },
        "diagnosis": {
            "top_path": "graph.source_guard",
            "ranked": [{"path_id": "graph.source_guard"}],
        },
        "lever_contract": {
            "lever_id": "enable-mha-device-graph",
            "top_path": "graph.source_guard",
            "expected_signals": [
                {"name": "graph_launch_us", "direction": "decrease", "min_ratio": 0.8}
            ],
            "source_files": ["infinilm/attention/mha_kv_cache.cpp"],
            "deployment_files": [],
            "correctness_gate": "all suite answers and streaming requests unchanged",
            "rollback": "restore guard return value and rebuild candidate pin",
            "wall_gate": {"profiled": False, "ratio_gt": 1.05},
        },
        "promote": {
            "one_path": "graph.source_guard",
            "may_apply": False,
            "must_reobserve_unprofiled": True,
        },
        "mechanism_check": {
            "status": "NOT_RUN",
            "before_profile_ref": "",
            "after_profile_ref": "",
            "results": [],
        },
    }


def test_valid_profile_v2_is_accepted():
    profile = validate_profile_v2(valid_profile())

    assert profile.window.alignment == "MATCHED"
    assert profile.workload.tensor_parallel == 4
    assert len(profile.ranks) == 2
    assert profile.timing.graph_replay_device_us == 700_000
    assert profile.source_map.symbols[0]["kind"] == "graph_guard"


def test_profile_v2_rejects_unmatched_window():
    document = valid_profile()
    document["window"]["alignment"] = "UNMATCHED"

    with pytest.raises(ContractValidationError) as excinfo:
        validate_profile_v2(document)

    assert any("alignment must be MATCHED" in error for error in excinfo.value.errors)


def test_profile_v2_rejects_window_duration_outside_tolerance():
    document = valid_profile()
    document["window"]["duration_ratio"] = 1.2

    with pytest.raises(ContractValidationError) as excinfo:
        validate_profile_v2(document)

    assert any("duration_ratio" in error for error in excinfo.value.errors)


def test_profile_v2_rejects_duplicate_or_partial_rank_views():
    document = valid_profile()
    document["ranks"][1]["rank"] = 0

    with pytest.raises(ContractValidationError) as excinfo:
        validate_profile_v2(document)

    assert any("rank is duplicated" in error for error in excinfo.value.errors)


def test_valid_request_v2_is_accepted():
    parsed = validate_promotion_request_v2(valid_request())

    assert parsed["lever_contract"].lever_id == "enable-mha-device-graph"
    assert parsed["mechanism_check"].status == "NOT_RUN"


def test_profile_v2_rejects_noncanonical_scheduler_signal():
    document = valid_profile()
    document["scheduler_signals"].append(
        {
            "family": "step_control",
            "name": "scheduler.step",
            "candidate_us": 100,
            "count": 1,
            "source": "naive-sidecar",
        }
    )

    with pytest.raises(ContractValidationError) as excinfo:
        validate_profile_v2(document)

    assert any(
        "must be a canonical InfiniLM span" in error for error in excinfo.value.errors
    )


def test_request_v2_rejects_missing_lever_contract():
    document = valid_request()
    del document["lever_contract"]

    with pytest.raises(ContractValidationError) as excinfo:
        validate_promotion_request_v2(document)

    assert "lever_contract is required" in excinfo.value.errors


def test_request_v2_rejects_contract_without_implementation_scope():
    document = valid_request()
    document["lever_contract"]["source_files"] = []

    with pytest.raises(ContractValidationError) as excinfo:
        validate_promotion_request_v2(document)

    assert any(
        "requires at least one source or deployment file" in error
        for error in excinfo.value.errors
    )


def test_request_v2_rejects_path_and_gate_mismatch():
    document = valid_request()
    document["diagnosis"]["top_path"] = "runtime.host_graph_command"

    with pytest.raises(ContractValidationError) as excinfo:
        validate_promotion_request_v2(document)

    assert any("must match diagnosis.top_path" in error for error in excinfo.value.errors)
