import json

import pytest

from infiniprofiler.adapters.hctracer import normalize


def trace(events, complete=True):
    return {"traceEvents": events} if complete else None


def write(tmp_path, name, doc):
    path = tmp_path / name
    if isinstance(doc, dict):
        path.write_text(json.dumps(doc), encoding="utf-8")
    else:
        path.write_text(doc, encoding="utf-8")
    return path


def test_normalizes_host_and_timing_without_scheduler_contract(tmp_path):
    path = write(
        tmp_path,
        "candidate.json",
        trace(
            [
                {"ph": "X", "name": "hcLaunchKernel", "dur": 200, "ts": 0, "pid": 1},
                {"ph": "X", "name": "hcMemcpy", "dur": 100, "ts": 200, "pid": 1},
                {
                    "ph": "X",
                    "name": "attention_kernel",
                    "cat": "kernel",
                    "dur": 150,
                    "ts": 300,
                    "pid": 1,
                },
                {
                    "ph": "X",
                    "name": "infinilm:batch_build",
                    "dur": 80,
                    "ts": 500,
                    "args": {"rank": 3},
                },
            ]
        ),
    )
    profile = normalize(path)

    assert profile.host("command").candidate == 200
    assert profile.host("api").candidate == 300
    assert profile.timing_attribution is not None
    assert profile.timing_attribution.gpu_execution_us > 0
    assert not hasattr(profile, "scheduler_signals")
    assert any("optional_marker_names_not_contract" in warning for warning in profile.warnings)
    assert "infinilm:batch_build" in " ".join(profile.warnings)


def test_trace_without_markers_still_emits_timing(tmp_path):
    path = write(
        tmp_path,
        "candidate.json",
        trace(
            [
                {
                    "ph": "X",
                    "name": "hcLaunchKernel",
                    "cat": "runtime",
                    "dur": 500,
                    "ts": 0,
                    "pid": 1,
                }
            ]
        ),
    )
    profile = normalize(path)

    assert profile.timing_attribution is not None
    assert profile.timing_attribution.api_submit_us == 500
    assert profile.host("gap").candidate == 500
    assert not any("scheduler_signals" in warning for warning in profile.warnings)
    assert not any(item["path_id"].startswith("scheduler.") for item in [])


def test_baseline_timing_is_stored_for_matched_ratios(tmp_path):
    baseline = write(
        tmp_path,
        "baseline.json",
        trace(
            [
                {
                    "ph": "X",
                    "name": "hcLaunchKernel",
                    "cat": "runtime",
                    "dur": 20,
                    "ts": 0,
                    "pid": 1,
                }
            ]
        ),
    )
    candidate = write(
        tmp_path,
        "candidate.json",
        trace(
            [
                {
                    "ph": "X",
                    "name": "hcLaunchKernel",
                    "cat": "runtime",
                    "dur": 40,
                    "ts": 0,
                    "pid": 1,
                }
            ]
        ),
    )
    profile = normalize(candidate, baseline_path=baseline)

    assert profile.host("api").baseline == 20
    assert profile.host("api").ratio == 2.0
    stored = profile.capture.provenance["baseline_timing_attribution"]
    assert stored["api_submit_us"] == 20
    assert profile.timing_attribution.api_submit_us == 40


def test_recovers_truncated_chrome_trace(tmp_path):
    complete = '{"traceEvents": [{"ph": "X", "name": "hcLaunchKernel", "dur": 20, "ts": 0, "pid": 1}'
    path = write(tmp_path, "truncated.json", complete)
    profile = normalize(path)

    assert any("recovered_truncated" in warning for warning in profile.warnings)
    assert profile.host("command").candidate == 20


def test_unrecoverable_trace_has_a_clear_error(tmp_path):
    path = write(tmp_path, "broken.json", "{not-json")
    with pytest.raises(ValueError, match="unrecoverable Chrome trace"):
        normalize(path)


def test_mctracer_normalize_exposes_graph_host_signal(tmp_path):
    path = write(
        tmp_path,
        "mctracer.json",
        trace(
            [
                {
                    "name": "mcThreadExchangeStreamCaptureMode",
                    "ph": "X",
                    "ts": 0,
                    "dur": 100,
                    "pid": 1,
                    "tid": 8,
                    "args": {"co_id": 1, "ret": 0},
                },
                {
                    "name": "Launch",
                    "ph": "s",
                    "cat": "Launch",
                    "ts": 200,
                    "pid": 1,
                    "tid": 201,
                    "id": 7,
                },
                {
                    "name": "Launch",
                    "ph": "f",
                    "cat": "Launch",
                    "ts": 500,
                    "pid": 3,
                    "tid": 0,
                    "id": 7,
                },
            ]
        ),
    )
    profile = normalize(path, source_name="mctracer")

    assert profile.host("graph") is not None
    assert profile.host("graph").candidate > 0
    assert profile.timing_attribution is not None
    assert profile.timing_attribution.graph_launch_us > 0
    assert profile.timing_attribution.graph_replay_device_us > 0
    assert (
        profile.timing_attribution.evidence_mode
        == "EXCLUSIVE_UNION_WITH_EXPLICIT_CORRELATION"
    )
