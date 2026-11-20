import json

from infiniprofiler.tools.sidecar import normalize
from infiniprofiler.tools.sidecar_producer import SidecarSpanRecorder


def test_producer_emits_parseable_spans(tmp_path):
    path = tmp_path / "spans.jsonl"
    recorder = SidecarSpanRecorder(path, rank=2)

    with recorder.span("infinilm.scheduler.step", tid=123):
        pass
    recorder.metric("tokens_per_iteration", 24)

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    profile = normalize(path)

    assert rows[0]["type"] == "span"
    assert rows[0]["rank"] == 2
    assert rows[1]["name"] == "tokens_per_iteration"
    assert profile.scheduler("step_control").count == 1
    assert profile.metric("tokens_per_iteration").candidate == 24
