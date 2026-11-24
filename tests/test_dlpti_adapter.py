import json
from pathlib import Path

import pytest

from infiniprofiler.adapters.dlpti import (
    detect_smoke_fingerprint,
    iter_activities,
    normalize,
)
from infiniprofiler.promotion.ranker import KERNEL_MATERIAL_RATIO_MIN, rank_paths

FIXTURES = Path(__file__).parent / "fixtures"


def write(tmp_path, name, doc):
    path = tmp_path / name
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_fixture_preserves_dlpti_source_and_host_families():
    candidate = FIXTURES / "dlpti_candidate.json"
    baseline = FIXTURES / "dlpti_baseline.json"
    profile = normalize(candidate, baseline_path=baseline, mode="wrap")

    assert profile.capture.source == "dlpti"
    assert profile.capture.mode == "wrap"
    assert not hasattr(profile, "scheduler_signals")
    assert any("no_scheduler_annotations" in w for w in profile.warnings)
    assert profile.host("driver") is not None
    assert profile.host("runtime") is not None
    assert profile.host("api") is not None
    assert profile.host("command") is not None
    # Candidate was built with longer spans than baseline → ratio > 1
    assert profile.host("api").baseline is not None
    assert profile.host("api").ratio is not None
    assert profile.host("api").ratio > 1.0


def test_smoke_fixture_does_not_emit_cu_bucket_metrics():
    candidate = FIXTURES / "dlpti_candidate.json"
    baseline = FIXTURES / "dlpti_baseline.json"
    profile = normalize(candidate, baseline_path=baseline, mode="wrap")
    assert any("smoke_fingerprint" in w for w in profile.warnings)
    assert profile.capture.provenance.get("smoke_fingerprint")
    assert profile.metrics == []
    assert not any(
        m.name.startswith("cu_bucket.") for m in profile.metrics
    )


def test_wrap_fixture_streams_hosts_and_cu_metrics():
    candidate = FIXTURES / "dlpti_wrap_candidate.json"
    baseline = FIXTURES / "dlpti_wrap_baseline.json"
    assert candidate.exists(), "trimmed wrap fixture missing"
    assert baseline.exists(), "trimmed wrap baseline fixture missing"

    # Streaming iterator must yield activities without full json.load path issues
    acts = list(iter_activities(candidate))
    assert len(acts) >= 100

    profile = normalize(candidate, baseline_path=baseline, mode="wrap")
    assert profile.capture.mode == "wrap"
    assert profile.capture.provenance.get("smoke_fingerprint") is None
    assert profile.host("kernel") is not None
    assert profile.host("memcpy") is not None
    cu = [m for m in profile.metrics if m.name.startswith("cu_bucket.")]
    assert cu, "expected non-empty CU bucket metrics from wrap fixture"
    assert any(m.name == "cu_bucket.flash_attention" for m in cu)
    assert any(m.ratio is not None for m in cu)


def test_maps_driver_runtime_command_kernel_memcpy_and_gap(tmp_path):
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
                    "kernel_name": "flash_attn_kernel",
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
    profile = normalize(path, mode="wrap")
    # start/end are ns → us: 200, 100, 50, 40, 10
    assert profile.host("driver").candidate == pytest.approx(200.0)
    assert profile.host("runtime").candidate == pytest.approx(100.0)
    assert profile.host("command").candidate == pytest.approx(250.0)  # launch + command
    assert profile.host("api").candidate == pytest.approx(300.0)
    assert profile.host("kernel").candidate == pytest.approx(40.0)
    assert profile.host("memcpy").candidate == pytest.approx(10.0)
    assert profile.host("graph").candidate == pytest.approx(1.0)  # untimed count proxy
    assert profile.host("gap").candidate == pytest.approx(300.0 + 250.0 - 40.0 - 10.0)
    assert profile.metric("cu_bucket.flash_attention") is not None
    assert profile.metric("cu_bucket.flash_attention").candidate == pytest.approx(40.0)


def test_rejects_non_native_export(tmp_path):
    path = write(tmp_path, "chrome.json", {"traceEvents": []})
    with pytest.raises(ValueError, match="DLPTI native export"):
        normalize(path)


def test_kernel_material_ranks_from_cu_bucket_ratio(tmp_path):
    cand = write(
        tmp_path,
        "cand.json",
        {
            "profiler_info": [
                {
                    "process_info": {
                        "cmdline": "dlpti_tools capture -- bash run.sh --backends infinilm"
                    }
                }
            ],
            "activities": [
                {
                    "kind": 56,
                    "name": "flash_attn_fwd",
                    "start": 0,
                    "end": 2_000_000,
                }
            ]
            * 250,
        },
    )
    base = write(
        tmp_path,
        "base.json",
        {
            "profiler_info": [
                {
                    "process_info": {
                        "cmdline": "dlpti_tools capture -- bash run.sh --backends vllm"
                    }
                }
            ],
            "activities": [
                {
                    "kind": 56,
                    "name": "flash_attn_fwd",
                    "start": 0,
                    "end": 1_000_000,
                }
            ]
            * 250,
        },
    )
    profile = normalize(cand, baseline_path=base, mode="wrap")
    metric = profile.metric("cu_bucket.flash_attention")
    assert metric is not None and metric.ratio is not None
    assert metric.ratio >= KERNEL_MATERIAL_RATIO_MIN
    ranked = rank_paths(profile, wall_ratio=0.135)
    assert ranked["top_path"] == "kernel.material"
    assert ranked["ranked"][0]["metric"] == "cu_bucket.flash_attention"


def test_detect_smoke_vector_add():
    reason = detect_smoke_fingerprint(
        {
            "profiler_info": [
                {
                    "process_info": {
                        "cmdline": ".../samples/cuda/vectorAdd/build/vector_add"
                    }
                }
            ]
        },
        activity_count=500,
        kernel_count=2,
    )
    assert reason and "vector_add" in reason
