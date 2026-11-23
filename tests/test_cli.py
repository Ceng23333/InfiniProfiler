import json

import pytest
from infiniprofiler.tools.process_sampler import _parse_status, _parse_stat_task
from infiniprofiler.tools import process_sampler

from infiniprofiler.cli import main
from tests.test_contracts import valid_profile, valid_request


def test_cli_normalizes_and_validates_sidecar(tmp_path, capsys):
    source = tmp_path / "scheduler.jsonl"
    output = tmp_path / "profile.json"
    source.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "metric",
                        "name": "batch_size",
                        "value": 6,
                        "window_id": "window-0",
                    },
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rc = main(
        [
            "normalize",
            "--source",
            "naive-sidecar",
            "--input",
            str(source),
            "--output",
            str(output),
        ]
    )
    captured = capsys.readouterr().out

    assert rc == 0
    assert "metrics=1" in captured
    assert "scheduler_signals" not in captured
    assert main(["validate", "--input", str(output)]) == 0
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["schema_version"] == "infiniprofiler-normalize-1"


def test_cli_detects_and_validates_profile_v2(tmp_path, capsys):
    path = tmp_path / "profile-v2.json"
    path.write_text(json.dumps(valid_profile()), encoding="utf-8")

    assert main(["validate", "--input", str(path)]) == 0
    captured = capsys.readouterr().out
    assert '"schema_version": "infiniprofiler-profile-2"' in captured
    assert '"alignment": "MATCHED"' in captured
    assert "api_submit_us" in captured


def test_cli_detects_and_validates_promotion_request_v2(tmp_path, capsys):
    path = tmp_path / "request-v2.json"
    path.write_text(json.dumps(valid_request()), encoding="utf-8")

    assert main(["validate", "--kind", "promotion-request", "--input", str(path)]) == 0
    assert '"top_path": "graph.source_guard"' in capsys.readouterr().out


def test_cli_emits_and_evaluates_mechanism_request_v2(tmp_path):
    from infiniprofiler.models import Capture, HostSignal, ProfileDocument, TimingAttributionFields
    from tests.test_mechanism import GRAPH_TIMING, LEVER, SOURCE_MAP, before_after

    document = ProfileDocument(capture=Capture(source="hctracer"))
    document.host_signals.append(
        HostSignal(family="graph", candidate=3100, baseline=100, ratio=31.0, source="hctracer")
    )
    document.timing_attribution = TimingAttributionFields.from_mapping(GRAPH_TIMING)
    profile = tmp_path / "profile.json"
    wall = tmp_path / "compare.json"
    lever = tmp_path / "lever.json"
    timing = tmp_path / "timing.json"
    source_map = tmp_path / "source-map.json"
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    request = tmp_path / "request-v2.json"
    updated = tmp_path / "request-v2-moved.json"
    profile.write_text(json.dumps(document.to_dict()), encoding="utf-8")
    wall.write_text(json.dumps({"ratio": 0.5}), encoding="utf-8")
    lever.write_text(json.dumps(LEVER), encoding="utf-8")
    timing.write_text(json.dumps(GRAPH_TIMING), encoding="utf-8")
    source_map.write_text(json.dumps(SOURCE_MAP), encoding="utf-8")
    before_doc, after_doc = before_after()
    before.write_text(json.dumps(before_doc), encoding="utf-8")
    after.write_text(json.dumps(after_doc), encoding="utf-8")

    rc = main(
        [
            "promotion-request-v2",
            "--profile",
            str(profile),
            "--wall-compare",
            str(wall),
            "--lever-contract",
            str(lever),
            "--window-id",
            "window-0",
            "--profile-ref",
            "profile/candidate.json",
            "--before-profile",
            str(before),
            "--after-profile",
            str(after),
            "--before-profile-ref",
            "profile/before.json",
            "--after-profile-ref",
            "profile/after.json",
            "--campaign-id",
            "campaign",
            "--baseline-pin",
            "frozen",
            "--candidate-pin",
            "candidate",
            "--iteration",
            "0",
            "--baseline-backend",
            "engine-frozen",
            "--candidate-backend",
            "engine-candidate",
            "--timing-attribution",
            str(timing),
            "--source-map",
            str(source_map),
            "--output",
            str(request),
        ]
    )

    assert rc == 0
    initial = json.loads(request.read_text(encoding="utf-8"))
    assert initial["schema_version"] == "promotion-request-2"
    assert initial["mechanism_check"]["status"] == "MOVED"
    assert main(
        [
            "evaluate-mechanism",
            "--request",
            str(request),
            "--before-profile",
            str(before),
            "--after-profile",
            str(after),
            "--before-profile-ref",
            "profile/before.json",
            "--after-profile-ref",
            "profile/after.json",
            "--output",
            str(updated),
        ]
    ) == 0
    assert json.loads(updated.read_text(encoding="utf-8"))["mechanism_check"]["status"] == "MOVED"


def test_cli_reports_contract_errors_without_traceback(tmp_path, capsys):
    path = tmp_path / "invalid.json"
    document = valid_profile()
    document["window"]["alignment"] = "UNMATCHED"
    path.write_text(json.dumps(document), encoding="utf-8")

    assert main(["validate", "--input", str(path)]) == 2
    captured = capsys.readouterr()
    assert "alignment must be MATCHED" in captured.err
    assert "Traceback" not in captured.err


def test_sample_process_requires_one_target(tmp_path):
    rc = main(
        [
            "sample-process",
            "--match",
            "not-a-live-process",
            "--duration-s",
            "0.1",
            "--output",
            str(tmp_path / "out.jsonl"),
        ]
    )
    assert rc == 2


def test_status_context_switch_parser(tmp_path):
    status = tmp_path / "status"
    status.write_text(
        "\n".join(
            [
                "Name: python",
                "voluntary_ctxt_switches: 12",
                "nonvoluntary_ctxt_switches: 3",
                "unknown: ignored",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert _parse_status(status) == (12, 3)


def test_clock_ticks_configuration_name_is_supported_on_linux():
    if not process_sampler.Path("/proc").exists():
        return
    assert process_sampler.os.sysconf("SC_CLK_TCK") > 0


def test_stat_task_parser_preserves_numeric_tid(tmp_path):
    stat = tmp_path / "stat"
    fields = ["1"] * 37
    fields[11] = "14"
    fields[12] = "15"
    fields[17] = "4"
    stat.write_text("42 (python3) " + " ".join(fields) + "\n", encoding="utf-8")
    parsed = _parse_stat_task(stat)

    assert parsed is not None
    tid, values = parsed
    assert tid == "42"
    assert values["utime"] == 14
    assert values["stime"] == 15
    assert values["num_threads"] == 4
