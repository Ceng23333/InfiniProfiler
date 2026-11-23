import json
import pytest

from infiniprofiler.models import (
    Capture,
    EngineMetric,
    HostSignal,
    ProfileDocument,
    TimingAttributionFields,
)
from infiniprofiler.promotion.autohelix import promotion_request, write_io_compat
from infiniprofiler.promotion.ranker import primary_wall_ratio, rank_paths
from infiniprofiler.tools.sidecar import normalize


def sidecar(path, candidate=True):
    rows = [
        {
            "type": "span",
            "name": "infinilm:schedule",
            "window_id": "window-0",
            "start_ns": 1_000_000,
            "end_ns": 2_000_000,
            "rank": 0,
        },
        {
            "type": "metric",
            "name": "tokens_per_iteration",
            "value": 20 if candidate else 40,
            "window_id": "window-0",
        },
        {
            "type": "metric",
            "name": "queue_depth",
            "value": 8 if candidate else 2,
            "window_id": "window-0",
        },
        "not-json",
    ]
    path.write_text(
        "\n".join(json.dumps(row) if isinstance(row, dict) else row for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def test_sidecar_normalizes_engine_metrics_only(tmp_path):
    candidate = sidecar(tmp_path / "candidate.jsonl")
    baseline = sidecar(tmp_path / "baseline.jsonl", candidate=False)
    profile = normalize(candidate, baseline_path=baseline)

    assert profile.metric("tokens_per_iteration").ratio == 0.5
    assert profile.metric("queue_depth").ratio == 4.0
    assert any("invalid_jsonl_line" in warning for warning in profile.warnings)
    assert any("sidecar_span_ignored_not_promotion_evidence" in warning for warning in profile.warnings)


def test_unknown_span_is_warning_not_fact(tmp_path):
    path = tmp_path / "spans.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "span",
                        "name": "scheduler.foo",
                        "window_id": "window-0",
                        "duration_us": 999,
                    }
                ),
                json.dumps(
                    {
                        "type": "metric",
                        "name": "queue_depth",
                        "value": 4,
                        "window_id": "window-0",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    profile = normalize(path)

    assert profile.metric("queue_depth").candidate == 4
    assert any(
        "sidecar_span_ignored_not_promotion_evidence:1:scheduler.foo" in warning
        for warning in profile.warnings
    )


def test_promotion_request_rejects_missing_sidecar_window(tmp_path):
    path = tmp_path / "spans.jsonl"
    path.write_text(
        json.dumps(
            {
                "type": "metric",
                "name": "queue_depth",
                "value": 4,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    profile = normalize(path)

    assert profile.capture.provenance["window_matched"] is False
    with pytest.raises(ValueError, match="no matched window"):
        promotion_request(
            profile,
            wall_compare={"ratio": 0.5},
            campaign_id="campaign",
            model="Qwen3-8B",
            host="metax-9-sg",
            device_track="mars",
            baseline_pin="vllm",
            candidate_pin="infinilm",
            iteration="0",
        )


def test_promotion_request_rejects_mismatched_sidecar_windows(tmp_path):
    candidate = tmp_path / "candidate.jsonl"
    baseline = tmp_path / "baseline.jsonl"
    candidate.write_text(
        json.dumps(
            {
                "type": "metric",
                "name": "queue_depth",
                "value": 4,
                "window_id": "candidate-window",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    baseline.write_text(
        json.dumps(
            {
                "type": "metric",
                "name": "queue_depth",
                "value": 2,
                "window_id": "baseline-window",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    profile = normalize(candidate, baseline_path=baseline)

    assert profile.capture.provenance["windows_matched"] is False
    with pytest.raises(ValueError, match="windows do not match"):
        promotion_request(
            profile,
            wall_compare={"ratio": 0.5},
            campaign_id="campaign",
            model="Qwen3-8B",
            host="metax-9-sg",
            device_track="mars",
            baseline_pin="vllm",
            candidate_pin="infinilm",
            iteration="0",
        )


def test_ranker_prioritizes_metric_only_packing_hint():
    profile = ProfileDocument(capture=Capture(source="test"))
    profile.metrics.append(
        EngineMetric(name="tokens_per_iteration", candidate=20, baseline=40, ratio=0.5)
    )
    result = rank_paths(profile, wall_ratio=0.5)

    assert result["top_path"] == "engine.batch_packing"
    assert result["ranked"][0]["lever"].startswith("tune admission/batch packing")


def test_ranker_sync_wait_path_from_matched_timing():
    profile = ProfileDocument(
        capture=Capture(
            source="test",
            provenance={
                "baseline_timing_attribution": {
                    "evidence_mode": "EXCLUSIVE_UNION_WITH_EXPLICIT_CORRELATION",
                    "api_submit_us": 10,
                    "gpu_execution_us": 100,
                    "launch_latency_us": 0,
                    "sync_wait_us": 100,
                    "graph_launch_us": 0,
                    "graph_replay_device_us": 0,
                    "cpu_gap_before_us": 0,
                    "cpu_gap_after_us": 0,
                    "unpaired_event_count": 0,
                    "paired_event_count": 1,
                }
            },
        ),
        timing_attribution=TimingAttributionFields(
            evidence_mode="EXCLUSIVE_UNION_WITH_EXPLICIT_CORRELATION",
            sync_wait_us=200,
            api_submit_us=10,
            gpu_execution_us=100,
        ),
    )
    result = rank_paths(profile, wall_ratio=0.5)

    assert result["top_path"] == "runtime.sync_wait"
    assert "scheduler" not in result["top_path"]


def test_runtime_gap_is_not_promoted_as_scheduler_effort():
    profile = ProfileDocument(capture=Capture(source="test"))
    profile.host_signals.append(
        HostSignal(family="gap", candidate=10, baseline=1, ratio=3.0, source="test")
    )
    result = rank_paths(profile, wall_ratio=0.5)

    assert result["top_path"] == "runtime.host_gap"
    assert "scheduler" not in result["top_path"]
    assert result["host_first_fired"] is True


def test_insufficient_evidence_cites_timing_not_spans():
    profile = ProfileDocument(capture=Capture(source="test"))
    result = rank_paths(profile, wall_ratio=1.0)

    assert result["top_path"] == "diagnosis.insufficient_evidence"
    assert "timing" in result["ranked"][0]["reasons"][0]


def test_promotion_request_stops_before_autohelix_applies_a_lever(tmp_path):
    profile = ProfileDocument(capture=Capture(source="hctracer"))
    profile.metrics.append(
        EngineMetric(name="tokens_per_iteration", candidate=20, baseline=40, ratio=0.4)
    )
    profile.timing_attribution = TimingAttributionFields(
        evidence_mode="EXCLUSIVE_UNION_WITH_EXPLICIT_CORRELATION",
        api_submit_us=10,
        gpu_execution_us=100,
    )
    compare = {"walls": [{"metric": "output_token_throughput", "ratio": 0.4, "primary": True}]}
    request = promotion_request(
        profile,
        wall_compare=compare,
        campaign_id="qwen3-8b-metax9",
        model="Qwen3-8B",
        host="metax-9-sg",
        device_track="mars",
        baseline_pin="vllm-frozen",
        candidate_pin="infinilm-candidate",
        iteration="0",
    )

    assert request["controller"] == "autohelix"
    assert request["stop_before_promote"] is True
    assert request["promote"]["may_apply"] is False
    assert request["promote"]["must_reobserve_unprofiled"] is True
    assert request["observation"]["timing_attribution"]["api_submit_us"] == 10
    assert request["diagnosis"]["ranked"][0]["path_id"] == "engine.batch_packing"


def test_io_compat_preserves_host_family_shape(tmp_path):
    profile = ProfileDocument(capture=Capture(source="hctracer"))
    profile.host_signals.append(
        HostSignal(family="command", candidate=20, baseline=10, ratio=2.0, source="hctracer")
    )
    paths = write_io_compat(
        profile=profile,
        wall_compare={"ratio": 0.5},
        out_dir=tmp_path / "profile",
    )
    compare = json.loads(paths["compare"].read_text(encoding="utf-8"))

    assert compare["host_signals"][0]["family"] == "command"
    assert compare["host_signals"][0]["ratio"] == 2.0
    assert "scheduler_signals" not in compare
    assert json.loads(paths["backlog"].read_text(encoding="utf-8"))["host_first_fired"] is True


def test_primary_wall_ratio_reads_common_shapes():
    assert primary_wall_ratio({"ratio": 0.7}) == 0.7
    assert primary_wall_ratio({"WALL_DECODE_RATIO": 0.8}) == 0.8
    assert primary_wall_ratio({"tok_s_ratio": 0.9}) == 0.9
    assert primary_wall_ratio({"walls": [{"ratio": 1.1}, {"ratio": 0.2, "primary": True}]}) == 0.2
