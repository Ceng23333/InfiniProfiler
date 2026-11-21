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


def test_normalizes_host_and_explicit_scheduler_markers(tmp_path):
    path = write(
        tmp_path,
        "candidate.json",
        trace(
            [
                {"ph": "X", "name": "hcLaunchKernel", "dur": 200},
                {"ph": "X", "name": "hcMemcpy", "dur": 100},
                {"ph": "X", "name": "attention_kernel", "cat": "kernel", "dur": 150},
                {"ph": "X", "name": "infinilm:batch_build", "dur": 80, "args": {"rank": 3}},
                {"ph": "X", "name": "infinilm:schedule", "dur": 40, "args": {"rank": 3}},
            ]
        ),
    )
    profile = normalize(path)

    assert profile.host("command").candidate == 200
    assert profile.host("api").candidate == 300
    assert profile.host("gap").candidate == 50
    assert {signal.name for signal in profile.scheduler_signals} == {
        "infinilm:batch_build",
        "infinilm:schedule",
    }
    assert profile.scheduler("batch_packing").ranks == [3]
    assert not any("no_scheduler_annotations" in warning for warning in profile.warnings)


def test_does_not_invent_scheduler_ranges_when_markers_absent(tmp_path):
    path = write(
        tmp_path,
        "candidate.json",
        trace([{"ph": "X", "name": "hcLaunchKernel", "dur": 500}]),
    )
    profile = normalize(path)

    assert profile.scheduler_signals == []
    assert profile.host("gap").candidate == 500
    assert any("no_scheduler_annotations" in warning for warning in profile.warnings)


def test_compares_matching_scheduler_markers(tmp_path):
    baseline = write(
        tmp_path,
        "baseline.json",
            trace([{"ph": "X", "name": "infinilm:schedule", "dur": 20}]),
    )
    candidate = write(
        tmp_path,
        "candidate.json",
            trace([{"ph": "X", "name": "infinilm:schedule", "dur": 40}]),
    )
    profile = normalize(candidate, baseline_path=baseline)
    signal = profile.scheduler("step_control")

    assert signal.baseline_us == 20
    assert signal.ratio == 2.0


def test_recovers_truncated_chrome_trace(tmp_path):
    complete = '{"traceEvents": [{"ph": "X", "name": "hcLaunchKernel", "dur": 20}'
    path = write(tmp_path, "truncated.json", complete)
    profile = normalize(path)

    assert any("recovered_truncated" in warning for warning in profile.warnings)
    assert profile.host("command").candidate == 20


def test_unrecoverable_trace_has_a_clear_error(tmp_path):
    path = write(tmp_path, "broken.json", "{not-json")
    with pytest.raises(ValueError, match="unrecoverable Chrome trace"):
        normalize(path)
