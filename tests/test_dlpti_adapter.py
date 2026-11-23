import json
from pathlib import Path

import pytest

from infiniprofiler.adapters.dlpti import normalize

FIXTURES = Path(__file__).parent / "fixtures"


def write(tmp_path, name, doc):
    path = tmp_path / name
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_fixture_preserves_dlpti_source_and_host_families():
    candidate = FIXTURES / "dlpti_candidate.json"
    baseline = FIXTURES / "dlpti_baseline.json"
    profile = normalize(candidate, baseline_path=baseline)

    assert profile.capture.source == "dlpti"
    assert profile.scheduler_signals == []
    assert any("no_scheduler_annotations" in w for w in profile.warnings)
    assert profile.host("driver") is not None
    assert profile.host("runtime") is not None
    assert profile.host("api") is not None
    assert profile.host("command") is not None
    # Candidate was built with longer spans than baseline → ratio > 1
    assert profile.host("api").baseline is not None
    assert profile.host("api").ratio is not None
    assert profile.host("api").ratio > 1.0


def test_maps_driver_runtime_command_and_kernel_gap(tmp_path):
    path = write(
        tmp_path,
        "cand.json",
        {
            "activities": [
                {
                    "kind": 1,
                    "name": "cuLaunchKernel",
                    "start": 1_000_000_000,
                    "end": 1_000_200_000,
                },
                {
                    "kind": 2,
                    "name": "cudaMemcpy",
                    "start": 1_000_000_000,
                    "end": 1_000_100_000,
                },
                {
                    "kind": 49,
                    "start": 1_000_000_000,
                    "end": 1_000_050_000,
                },
                {
                    "kind": 56,
                    "kernel_name": "vectorAdd",
                    "start": 1_000_000_000,
                    "end": 1_000_040_000,
                },
                {
                    "kind": 50,
                    "start": 1_000_000_000,
                    "end": 1_000_010_000,
                },
                {"kind": 34, "node_name": "graph_node"},
            ]
        },
    )
    profile = normalize(path)
    # start/end are ns → us: 200, 100, 50, 40, 10
    assert profile.host("driver").candidate == pytest.approx(200.0)
    assert profile.host("runtime").candidate == pytest.approx(100.0)
    assert profile.host("command").candidate == pytest.approx(250.0)  # launch + command
    assert profile.host("api").candidate == pytest.approx(300.0)
    assert profile.host("graph").candidate == pytest.approx(1.0)  # untimed count proxy
    assert profile.host("gap").candidate == pytest.approx(300.0 + 250.0 - 40.0 - 10.0)


def test_rejects_non_native_export(tmp_path):
    path = write(tmp_path, "chrome.json", {"traceEvents": []})
    with pytest.raises(ValueError, match="DLPTI native export"):
        normalize(path)
