import json

import pytest

from infiniprofiler.models import Capture, HostSignal, ProfileDocument
from infiniprofiler.promotion.mechanism import (
    MechanismError,
    apply_mechanism_evaluation,
    build_promotion_request_v2,
    evaluate_mechanism,
)
from tests.test_contracts import valid_profile


SOURCE_MAP = {
    "symbols": [{"kind": "graph_guard", "file": "engine/graph.cpp"}],
    "providers": [],
}

GRAPH_TIMING = {
    "evidence_mode": "EXCLUSIVE_UNION_WITH_EXPLICIT_CORRELATION",
    "graph_launch_us": 1000,
    "graph_replay_device_us": 7000,
}

LEVER = {
    "lever_id": "narrow-mha-graph-guard",
    "top_path": "graph.source_guard",
    "expected_signals": [
        {"name": "graph_launch_us", "direction": "decrease", "min_ratio": 0.8}
    ],
    "source_files": ["engine/attention/graph.cpp"],
    "deployment_files": [],
    "correctness_gate": "streaming responses and suite accuracy unchanged",
    "rollback": "restore the graph guard",
    "wall_gate": {"profiled": False, "ratio_gt": 1.05},
}


def profile():
    document = ProfileDocument(capture=Capture(source="hctracer"))
    document.host_signals.append(
        HostSignal(family="graph", candidate=3100, baseline=100, ratio=31.0, source="hctracer")
    )
    return document


def before_after(launch_before=8000, launch_after=4000):
    before = valid_profile()
    after = valid_profile()
    before["timing_attribution"]["graph_launch_us"] = launch_before
    after["timing_attribution"]["graph_launch_us"] = launch_after
    return before, after


def build_request(after=None):
    before, after_doc = before_after(launch_after=4000 if after is None else after)
    return build_promotion_request_v2(
        profile(),
        wall_compare={"walls": [{"ratio": 0.5, "primary": True}]},
        lever_contract=LEVER,
        window_id="window-0",
        campaign_id="qwen3-8b-metax9-sg",
        model="Qwen3-8B",
        host="metax-9-sg",
        device_track="mars",
        baseline_pin="engine-frozen",
        candidate_pin="engine-candidate",
        iteration="0",
        profile_ref="profile/candidate.json",
        before_profile_ref="profile/before.json",
        after_profile_ref="profile/after.json" if after is not None else "",
        before=before,
        after=after_doc if after is not None else None,
        baseline_backend="engine-frozen",
        candidate_backend="engine-candidate",
        graph_timing=GRAPH_TIMING,
        graph_source=SOURCE_MAP,
    )


def test_initial_v2_request_has_required_contract_and_does_not_promote():
    request = build_request(after=None)

    assert request["schema_version"] == "promotion-request-2"
    assert request["stop_before_promote"] is True
    assert request["promote"]["may_apply"] is False
    assert request["gate"]["profiled_result_can_promote"] is False
    assert request["gate"]["surpass_threshold"] == 1.05
    assert request["mechanism_check"]["status"] == "NOT_RUN"
    assert request["campaign"]["baseline_backend"] == "engine-frozen"
    assert request["campaign"]["candidate_backend"] == "engine-candidate"


def test_favorable_expected_signal_classifies_moved():
    request = build_request(after=4000)

    assert request["mechanism_check"]["status"] == "MOVED"
    result = request["mechanism_check"]["results"][0]
    assert result["before"] == 8000
    assert result["after"] == 4000
    assert result["ratio"] == 0.5
    assert result["status"] == "PASSED"


def test_unfavorable_expected_signal_classifies_regressed():
    request = build_request(after=12000)

    assert request["mechanism_check"]["status"] == "REGRESSED"
    assert request["mechanism_check"]["results"][0]["status"] == "REGRESSED"


def test_missing_expected_signal_is_inconclusive():
    before, after = before_after()
    del before["timing_attribution"]["graph_launch_us"]
    del after["timing_attribution"]["graph_launch_us"]
    evaluation = evaluate_mechanism(
        LEVER,
        before=before,
        after=after,
        before_profile_ref="before.json",
        after_profile_ref="after.json",
    )

    assert evaluation.status == "INCONCLUSIVE"
    assert evaluation.results[0]["status"] == "INCONCLUSIVE"


def test_apply_mechanism_revalidates_request(tmp_path):
    initial = build_request(after=None)
    before, after = before_after()
    updated = apply_mechanism_evaluation(
        initial,
        before=before,
        after=after,
        before_profile_ref="profile/before.json",
        after_profile_ref="profile/after.json",
    )

    assert updated["mechanism_check"]["status"] == "MOVED"
    assert updated["stop_before_promote"] is True


def test_observed_top_path_must_match_lever(tmp_path):
    lever = dict(LEVER)
    lever["top_path"] = "scheduler.batch_packing"
    with pytest.raises(MechanismError, match="does not match observed"):
        build_promotion_request_v2(
            profile(),
            wall_compare={"ratio": 0.5},
            lever_contract=lever,
            window_id="window-0",
            campaign_id="campaign",
            model="Qwen3-8B",
            host="metax-9-sg",
            device_track="mars",
            baseline_pin="frozen",
            candidate_pin="candidate",
            iteration="0",
            profile_ref="profile.json",
            before_profile_ref="before.json",
            graph_timing=GRAPH_TIMING,
            graph_source=SOURCE_MAP,
        )
