import json

from infiniprofiler.models import ProfileDocument, Capture, EngineMetric, HostSignal
from infiniprofiler.promotion.autohelix import promotion_request, write_io_compat
from infiniprofiler.promotion.ranker import primary_wall_ratio, rank_paths
from infiniprofiler.tools.sidecar import normalize


def sidecar(path, candidate=True):
    rows = [
        {
            "type": "span",
            "name": "schedule.step",
            "start_ns": 1_000_000,
            "end_ns": 2_000_000,
            "rank": 0,
        },
        {
            "type": "span",
            "name": "batch.build",
            "duration_us": 100,
            "rank": 1,
        },
        {"type": "metric", "name": "tokens_per_iteration", "value": 20 if candidate else 40},
        {"type": "metric", "name": "queue_depth", "value": 8 if candidate else 2},
        "not-json",
    ]
    path.write_text(
        "\n".join(json.dumps(row) if isinstance(row, dict) else row for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def test_sidecar_normalizes_scheduler_and_engine_ratios(tmp_path):
    candidate = sidecar(tmp_path / "candidate.jsonl")
    baseline = sidecar(tmp_path / "baseline.jsonl", candidate=False)
    profile = normalize(candidate, baseline_path=baseline)

    assert profile.scheduler("step_control").candidate_us == 1000
    assert profile.scheduler("batch_packing").ratio == 1.0
    assert profile.scheduler("step_control").ranks == [0]
    assert profile.scheduler("batch_packing").ranks == [1]
    assert profile.metric("tokens_per_iteration").ratio == 0.5
    assert profile.metric("queue_depth").ratio == 4.0
    assert any("invalid_jsonl_line" in warning for warning in profile.warnings)


def test_ranker_prioritizes_explicit_scheduler_evidence():
    profile = ProfileDocument(capture=Capture(source="test"))
    profile.metrics.append(
        EngineMetric(name="tokens_per_iteration", candidate=20, baseline=40, ratio=0.5)
    )
    result = rank_paths(profile, wall_ratio=0.5)

    assert result["top_path"] == "scheduler.batch_packing"
    assert result["host_first_fired"] is False
    assert result["ranked"][0]["lever"].startswith("tune admission/batch packing")


def test_runtime_gap_is_not_promoted_as_scheduler_effort():
    profile = ProfileDocument(capture=Capture(source="test"))
    profile.host_signals.append(
        HostSignal(family="gap", candidate=10, baseline=1, ratio=3.0, source="test")
    )
    result = rank_paths(profile, wall_ratio=0.5)

    assert result["top_path"] == "runtime.host_graph_command"
    assert result["host_first_fired"] is True


def test_promotion_request_stops_before_autohelix_applies_a_lever(tmp_path):
    profile = ProfileDocument(capture=Capture(source="hctracer"))
    profile.metrics.append(
        EngineMetric(name="tokens_per_iteration", candidate=20, baseline=40, ratio=0.4)
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
    assert request["diagnosis"]["ranked"][0]["reasons"][0].endswith("candidate/baseline=0.400")


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
    assert json.loads(paths["backlog"].read_text(encoding="utf-8"))["host_first_fired"] is True


def test_primary_wall_ratio_reads_common_shapes():
    assert primary_wall_ratio({"ratio": 0.7}) == 0.7
    assert primary_wall_ratio({"WALL_DECODE_RATIO": 0.8}) == 0.8
    assert primary_wall_ratio({"walls": [{"ratio": 1.1}, {"ratio": 0.2, "primary": True}]}) == 0.2
