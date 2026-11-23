import json

from infiniprofiler.tools.sidecar import normalize
from infiniprofiler.tools.sidecar_producer import NullSpanRecorder, SidecarSpanRecorder


def test_producer_emits_metrics_and_ignores_spans_for_promotion(tmp_path):
    path = tmp_path / "spans.jsonl"
    recorder = SidecarSpanRecorder(path, rank=2, window_id="window-0")

    with recorder.span("optional-marker", tid=123, iteration=7):
        pass
    recorder.metric("tokens_per_iteration", 24)

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    profile = normalize(path)

    assert rows[0]["type"] == "span"
    assert rows[0]["span_id"] == 1
    assert rows[0]["window_id"] == "window-0"
    assert rows[0]["iteration"] == 7
    assert rows[0]["rank"] == 2
    assert rows[1]["name"] == "tokens_per_iteration"
    assert profile.metric("tokens_per_iteration").candidate == 24
    assert any(
        "sidecar_span_provenance_only_not_promotion_evidence" in warning
        for warning in profile.warnings
    )
    assert profile.capture.provenance.get("engine_ranges_contract") == (
        "provenance_only_not_scheduler_signals"
    )
    ranges = profile.capture.provenance.get("engine_ranges") or []
    assert any(item.get("name") == "optional-marker" for item in ranges)


def test_null_producer_writes_nothing(tmp_path):
    disabled = NullSpanRecorder()
    with disabled.span("optional-marker", iteration=1):
        pass
    disabled.metric("tokens_per_iteration", 42)

    assert list(tmp_path.iterdir()) == []
