import json

import pytest
from infiniprofiler.tools.process_sampler import _parse_status
from infiniprofiler.tools import process_sampler

from infiniprofiler.cli import main


def test_cli_normalizes_and_validates_sidecar(tmp_path, capsys):
    source = tmp_path / "scheduler.jsonl"
    output = tmp_path / "profile.json"
    source.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "span",
                        "name": "scheduler.step",
                        "duration_us": 25,
                    }
                ),
                json.dumps(
                    {"type": "metric", "name": "batch_size", "value": 6}
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
    assert "scheduler_signals=1" in captured
    assert main(["validate", "--input", str(output)]) == 0
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["schema_version"] == "infiniprofiler-profile-1"


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
