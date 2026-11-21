import json
import pytest

from infiniprofiler.tools.sidecar import normalize
from infiniprofiler.tools.sidecar_producer import NullSpanRecorder, SidecarSpanRecorder


def test_producer_emits_parseable_spans(tmp_path):
    path = tmp_path / "spans.jsonl"
    recorder = SidecarSpanRecorder(path, rank=2, window_id="window-0")

    with recorder.span("infinilm:schedule", tid=123, iteration=7):
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
    assert profile.scheduler("step_control").count == 1
    assert profile.metric("tokens_per_iteration").candidate == 24


def test_producer_rejects_noncanonical_scheduler_span(tmp_path):
    recorder = SidecarSpanRecorder(tmp_path / "spans.jsonl", rank=2, window_id="window-0")
    with pytest.raises(ValueError, match="canonical"):
        with recorder.span("scheduler.step"):
            pass


def test_null_producer_writes_nothing(tmp_path):
    disabled = NullSpanRecorder()
    with disabled.span("infinilm:schedule", iteration=1):
        pass
    disabled.metric("tokens_per_iteration", 42)

    assert list(tmp_path.iterdir()) == []
