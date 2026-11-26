import json

import pytest

from infiniprofiler.adapters.mctracer import (
    ACTIVITY_KINDS,
    SEMANTIC_BUCKETS,
    analyze_trace,
    bucket_keys,
    classify_semantic,
    compare_gate,
)
from infiniprofiler.timeline import attribute_trace

NS = 1_700_000_000_000_000_000  # epoch-ns style base


def _x(name, ts_ns, dur_ns, *, pid=1, tid=1, cat=None, args=None):
    event = {"name": name, "ph": "X", "ts": NS + ts_ns, "dur": dur_ns, "pid": pid, "tid": tid}
    if cat is not None:
        event["cat"] = cat
    if args:
        event["args"] = args
    return event


@pytest.fixture
def trace_path(tmp_path):
    events = [
        _x("flashAttentionDecode", 0, 1000, pid=1, tid=10),
        _x("flashAttentionDecode", 500, 1000, pid=2, tid=10),
        _x("mcblas__Mck_bf16gemm", 2000, 500),
        _x("mcclKernel_AllReduce", 3000, 400),
        _x("rmsnormKernel", 3500, 200),
        _x("SwiGLUCuda<__maca_bfloat16, 1024u>(__maca_bfloat16*)", 4200, 200),
        _x("embeddingKernel", 4400, 100),
        _x("ropeThreadPerItemKernel", 4600, 100),
        _x("triton_poi_fused_add_mul_sub_5", 4800, 150),
        _x("triton_poi_fused_mul_silu_1", 5000, 120),
        _x("D2H", 5000, 300, cat="memcpy"),
        _x("fillKernel", 6000, 100),
        _x("rearrange_unit_float1_block_1", 6200, 50),
        _x("unknown_dev", 7000, 200, cat="4", args={"device_id": 0, "stream_id": 1}),
        _x("mcLaunchKernel", 50, 80, args={"_ip_role": "host"}),
        {"name": "Launch", "cat": "Launch", "ph": "s", "id": 7, "ts": NS + 100, "pid": 1, "tid": 1},
        {"name": "Launch", "cat": "Launch", "ph": "f", "id": 7, "ts": NS + 400, "pid": 3, "tid": 1},
    ]
    path = tmp_path / "tracer_out.json"
    path.write_text(json.dumps({"traceEvents": events}))
    return path


def test_taxonomy_layers():
    assert ACTIVITY_KINDS == ("kernel", "memcpy", "memset", "sync", "host_api", "graph")
    assert "mlp_activation" in SEMANTIC_BUCKETS and "embedding" in SEMANTIC_BUCKETS
    assert tuple(bucket_keys()) == ACTIVITY_KINDS


def test_semantic_helper_not_gate_contract():
    # Layer 2 is a non-contract helper for ad-hoc inspection only.
    assert classify_semantic("flashAttentionDecodeHd128SplitKvCta", None) == ("kernel", "attention")
    assert classify_semantic("mcblas__Mck_bf16gemm_tn", None) == ("kernel", "gemm")
    assert classify_semantic("mcclKernel_AllReduce_FC4_Sum", None) == ("kernel", "allreduce")
    assert classify_semantic("rmsnormKernel", None) == ("kernel", "residual_rmsnorm")
    assert classify_semantic("triton_poi_fused_add_mul_sub_5", None) == ("kernel", "residual_rmsnorm")
    assert classify_semantic("SwiGLUCuda<__maca_bfloat16, 1024u>(__maca_bfloat16*)", None) == ("kernel", "mlp_activation")
    assert classify_semantic("triton_poi_fused_mul_silu_1", None) == ("kernel", "mlp_activation")
    assert classify_semantic("embeddingKernel", None) == ("kernel", "embedding")
    assert classify_semantic("ropeThreadPerItemKernel", None) == ("kernel", "rope")
    assert classify_semantic("rearrange_unit_float1_block_1", None) == ("kernel", "kv_cache_reshape")
    assert classify_semantic("cudaMemcpyAsync", "memcpy") == ("memcpy", "")
    assert classify_semantic("mcMemsetAsync", None) == ("memset", "")
    assert classify_semantic("sampleTokensKernel", None) == ("kernel", "sampling")
    assert classify_semantic("totally_unknown", None) == ("kernel", "other")


def test_analyze_trace_units_and_union(trace_path):
    report = analyze_trace(trace_path)
    kernel = next(b for b in report["buckets"] if b["bucket"] == "kernel")
    assert kernel["activity_kind"] == "kernel"
    assert kernel["count"] == 15
    assert "semantic" not in kernel
    memset = next(b for b in report["buckets"] if b["bucket"] == "memset")
    assert memset["activity_kind"] == "memset"
    assert memset["coverage_us"] == pytest.approx(0.1)
    assert "4" in report["raw_categories"]["counts"]
    assert report["device"]["coverage_us"] < report["trace_span_us"]
    assert report["host_submit"]["idle_us"] > 0


def test_launch_flow_not_synthetic_device(trace_path, tmp_path):
    db = tmp_path / "timeline.sqlite3"
    attribution = attribute_trace(trace_path, database_path=db)
    assert "launch_flow_device_duration_not_synthetic" in attribution.warnings
    import sqlite3

    conn = sqlite3.connect(str(db))
    rows = conn.execute(
        "SELECT COUNT(*) FROM events WHERE name = 'Launch' AND category IN "
        "('gpu_execution','graph_device')"
    ).fetchone()
    assert rows[0] == 0


def test_compare_gate_fields(trace_path):
    cand = analyze_trace(trace_path)
    base = analyze_trace(trace_path)
    gate = compare_gate(cand, base)
    fam = gate["families"]["kernel"]
    assert fam["activity_kind"] == "kernel"
    assert "semantic" not in fam
    assert set(fam) >= {
        "candidate_us",
        "baseline_us",
        "ratio",
        "candidate_share",
        "baseline_share",
        "share_gap_pp",
        "magnitude_gap_us",
    }
    assert set(gate["families"]) <= set(ACTIVITY_KINDS)
    assert gate["families"]["memcpy"]["activity_kind"] == "memcpy"
    overall = gate["overall"]
    assert overall["gap_ratio"] == pytest.approx(1.0)
