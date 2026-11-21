import json
import os
import stat
from pathlib import Path
from typing import Dict, List

import pytest

from infiniprofiler.capture import (
    CaptureError,
    CaptureSpec,
    capture_campaign,
    compare_capture_manifests,
    merge_capture,
)


def capture_spec(**overrides) -> CaptureSpec:
    values: Dict[str, object] = {
        "schema_version": "infiniprofiler-capture-spec-1",
        "role": "candidate",
        "window_id": "window-0",
        "campaign_id": "qwen3-8b-metax9-sg",
        "model": "Qwen3-8B",
        "baseline_pin": "vllm-frozen",
        "candidate_pin": "infinilm-candidate",
        "tensor_parallel": 2,
        "wall_workload_id": "decode-256",
        "wall_compare_ref": "suite/compare.json",
        "warmup_excluded": True,
        "client_command": "true",
        "concurrency": 8,
        "prompt_lengths": [512],
        "output_lengths": [256],
        "clock_policy": "operator_observed",
        "profiles_source_hash": "sha256:candidate",
        "rank_pids": {"0": 1000, "1": 1001},
    }
    values.update(overrides)
    return CaptureSpec.from_dict(values)  # type: ignore[arg-type]


def fake_tracer(tmp_path: Path, *, succeed: bool = True) -> Path:
    path = tmp_path / ("tracer_ok" if succeed else "tracer_fail")
    if succeed:
        body = '''#!/usr/bin/env python3
import json, sys
from pathlib import Path
raw = Path("raw")
raw.mkdir(parents=True, exist_ok=True)
document = {"traceEvents": []}
text = json.dumps(document)
text += " " * max(0, 65536 - len(text))
(raw / "tracer_out-0.json").write_text(text, encoding="utf-8")
sys.stdin.buffer.read(1)
sys.stdin.buffer.read()
'''
    else:
        body = "#!/usr/bin/env python3\nimport sys\n"
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def tracer_command(path: Path):
    return lambda rank, pid: [str(path)]


def immediate_failure_command(rank: int, pid: int) -> List[str]:
    return ["/usr/bin/false"] if rank == 1 else ["/bin/true"]


def test_spec_rejects_duplicate_pids_and_partial_tp():
    with pytest.raises(CaptureError) as failed:
        capture_spec(rank_pids={"0": 1000, "1": 1000})
    assert "multiple ranks to pid 1000" in str(failed.value)

    with pytest.raises(CaptureError) as failed:
        capture_spec(rank_pids={"0": 1000})
    assert "exactly 2 ranks" in str(failed.value)


def test_successful_two_rank_capture_writes_shared_window_manifest(tmp_path):
    tracer = fake_tracer(tmp_path)
    capture = capture_campaign(
        capture_spec(),
        tmp_path / "run",
        ready_seconds=0.4,
        stop_wait_seconds=2.0,
        tracer_command=tracer_command(tracer),
    )

    assert capture.status == "SUCCEEDED"
    assert capture.started_epoch_ns < capture.ended_epoch_ns
    assert [rank.status for rank in capture.ranks] == ["SUCCEEDED", "SUCCEEDED"]
    manifest = json.loads((tmp_path / "run" / "window_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "SUCCEEDED"
    assert len(manifest["ranks"]) == 2


def test_failed_rank_aborts_transactionally(tmp_path):
    good = fake_tracer(tmp_path, succeed=True)

    def mixed_command(rank: int, pid: int) -> List[str]:
        return [str(good)] if rank == 0 else immediate_failure_command(rank, 1001)

    with pytest.raises(CaptureError, match="rank attach failed"):
        capture_campaign(
            capture_spec(),
            tmp_path / "run",
            ready_seconds=0.2,
            stop_wait_seconds=2.0,
            tracer_command=mixed_command,
        )

    manifest = json.loads((tmp_path / "run" / "window_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "FAILED_RANK_ATTACH"
    statuses = {rank["rank"]: rank["status"] for rank in manifest["ranks"]}
    assert statuses == {0: "STOPPED_DURING_FAILURE", 1: "FAILED_ATTACH"}


def test_client_failure_aborts_and_refuses_merge(tmp_path):
    tracer = fake_tracer(tmp_path)
    run_dir = tmp_path / "run"

    with pytest.raises(CaptureError, match="client window failed"):
        capture_campaign(
            capture_spec(client_command="false"),
            run_dir,
            ready_seconds=0.1,
            stop_wait_seconds=2.0,
            tracer_command=tracer_command(tracer),
        )

    manifest = json.loads((run_dir / "window_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "CLIENT_WINDOW_FAILED"
    assert all(rank["status"] == "CLIENT_WINDOW_FAILED" for rank in manifest["ranks"])
    with pytest.raises(CaptureError, match="refusing to merge"):
        merge_capture(run_dir)


def test_capture_manifest_comparison_matches_identity_and_window(tmp_path):
    manifest = {
        "schema_version": "infiniprofiler-capture-manifest-1",
        "status": "SUCCEEDED",
        "duration_ns": 1_000_000,
        "spec": capture_spec().to_dict(),
        "ranks": [
            {"rank": 0, "status": "SUCCEEDED"},
            {"rank": 1, "status": "SUCCEEDED"},
        ],
    }
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(json.dumps(manifest), encoding="utf-8")
    candidate_manifest = dict(manifest)
    candidate_manifest["duration_ns"] = 1_020_000
    candidate.write_text(json.dumps(candidate_manifest), encoding="utf-8")

    result = compare_capture_manifests(baseline, candidate)
    assert result["status"] == "MATCHED"
    assert result["duration_ratio"] == pytest.approx(1.02)


def test_capture_manifest_comparison_rejects_workload_change(tmp_path):
    manifest = {
        "schema_version": "infiniprofiler-capture-manifest-1",
        "status": "SUCCEEDED",
        "duration_ns": 1_000_000,
        "spec": capture_spec().to_dict(),
        "ranks": [
            {"rank": 0, "status": "SUCCEEDED"},
            {"rank": 1, "status": "SUCCEEDED"},
        ],
    }
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(json.dumps(manifest), encoding="utf-8")
    candidate_manifest = dict(manifest)
    candidate_manifest["spec"] = dict(manifest["spec"])
    candidate_manifest["spec"]["profiles_source_hash"] = "sha256:changed"
    candidate.write_text(json.dumps(candidate_manifest), encoding="utf-8")

    result = compare_capture_manifests(baseline, candidate)
    assert result["status"] == "MISMATCHED"
    assert any("profiles_source_hash" in error for error in result["errors"])


def test_merge_capture_normalizes_required_ranks(tmp_path):
    tracer = fake_tracer(tmp_path)
    run_dir = tmp_path / "run"
    capture_campaign(
        capture_spec(),
        run_dir,
        ready_seconds=0.4,
        stop_wait_seconds=2.0,
        tracer_command=tracer_command(tracer),
    )
    merged = merge_capture(run_dir)

    assert merged["status"] == "COMPLETE"
    assert merged["rank_count"] == 2
    assert [rank["rank"] for rank in merged["ranks"]] == [0, 1]
    assert all(rank["timing_attribution"]["event_count"] == 0 for rank in merged["ranks"])
    assert (run_dir / "capture_merge.json").is_file()


def test_merge_rejects_partial_capture(tmp_path):
    tracer = fake_tracer(tmp_path)
    run_dir = tmp_path / "run"
    with pytest.raises(CaptureError):
        capture_campaign(
            capture_spec(rank_pids={"0": 1000, "1": 999999}),
            run_dir,
            ready_seconds=0.1,
            stop_wait_seconds=0.2,
            tracer_command=immediate_failure_command,
        )
    # The failed attach still wrote a transactional manifest, so merge must stop.
    with pytest.raises(CaptureError, match="refusing to merge"):
        merge_capture(run_dir)
