import json

import pytest

from infiniprofiler.models import Capture, HostSignal, ProfileDocument
from infiniprofiler.promotion.ranker import rank_paths
from infiniprofiler.source_map import (
    SourceMapError,
    build_source_map,
    default_rules,
    load_rules,
)
from infiniprofiler.cli import main


def graph_timing(*, evidence_mode="EXCLUSIVE_UNION_WITH_EXPLICIT_CORRELATION", launch=100, replay=700):
    return {
        "evidence_mode": evidence_mode,
        "graph_launch_us": launch,
        "graph_replay_device_us": replay,
    }


def source_document(tmp_path):
    source_root = tmp_path / "engine"
    graph_file = source_root / "attention" / "graph.cpp"
    graph_file.parent.mkdir(parents=True)
    graph_file.write_text(
        "\n".join(
            [
                "#include <cstdint>",
                "namespace engine {",
                "bool MhaKVCache::supports_device_graph_capture() { return false; }",
                "}  // namespace engine",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    provider_file = source_root / "attention" / "provider.py"
    provider_file.write_text("provider = 'flash_attn'\n", encoding="utf-8")
    return source_root


def test_source_map_is_engine_neutral_and_records_exact_lines(tmp_path):
    source_root = source_document(tmp_path)
    symbol = "MhaKVCache::supports_device_graph_capture"
    document = build_source_map(
        source_root,
        source_pin="engine-candidate",
        rules=default_rules(),
        required_symbols=[symbol],
    )

    assert document["schema_version"] == "source-map-1"
    assert len(document["source_root_hash"]) == 64
    assert document["scanned_file_count"] == 2
    symbol_entry = next(item for item in document["symbols"] if item["symbol"] == symbol)
    assert symbol_entry["file"] == "attention/graph.cpp"
    assert symbol_entry["line_start"] == 3
    assert symbol_entry["kind"] == "graph_guard"
    assert any(provider["provider"] == "flash_attention" for provider in document["providers"])


def test_missing_required_symbol_fails_closed(tmp_path):
    source_root = source_document(tmp_path)
    with pytest.raises(SourceMapError, match="required symbols not found"):
        build_source_map(
            source_root,
            source_pin="engine-candidate",
            required_symbols=["NotPresent::symbol"],
        )


def test_required_symbol_comment_is_not_code_provenance(tmp_path):
    source_root = tmp_path / "engine"
    source_root.mkdir()
    source_file = source_root / "graph.py"
    source_file.write_text(
        "# MhaKVCache::supports_device_graph_capture\n",
        encoding="utf-8",
    )
    with pytest.raises(SourceMapError, match="required symbols not found"):
        build_source_map(
            source_root,
            source_pin="candidate",
            required_symbols=["MhaKVCache::supports_device_graph_capture"],
        )


def test_cli_builds_source_map(tmp_path):
    source_root = source_document(tmp_path)
    output = tmp_path / "source-map.json"

    rc = main(
        [
            "build-source-map",
            "--source-root",
            str(source_root),
            "--source-pin",
            "engine-candidate",
            "--rules",
            "examples/source-map-rules.json",
            "--required-symbol",
            "MhaKVCache::supports_device_graph_capture",
            "--output",
            str(output),
        ]
    )
    document = json.loads(output.read_text(encoding="utf-8"))

    assert rc == 0
    assert document["schema_version"] == "source-map-1"
    assert output.is_file()


def test_load_rules_validates_schema(tmp_path):
    path = tmp_path / "rules.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "source-map-rules-1",
                "rules": [
                    {
                        "kind": "provider",
                        "provider": "custom_provider",
                        "patterns": ["custom_provider"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    rules = load_rules(path)

    assert rules[0].kind == "provider"
    assert rules[0].provider == "custom_provider"


def graph_profile():
    profile = ProfileDocument(capture=Capture(source="hctracer"))
    profile.host_signals.append(
        HostSignal(
            family="graph",
            candidate=3100,
            baseline=100,
            ratio=31.0,
            source="hctracer",
        )
    )
    return profile


def test_high_graph_ratio_alone_does_not_select_graph_or_runtime_path():
    result = rank_paths(graph_profile(), wall_ratio=0.5)

    assert result["top_path"] == "diagnosis.insufficient_evidence"
    assert result["graph_evidence"]["selected"] is False
    assert not any(item["path_id"] == "runtime.host_graph_command" for item in result["ranked"])


def test_source_guard_and_paired_graph_timing_beats_generic_runtime_path():
    profile = graph_profile()
    profile.host_signals.append(
        HostSignal(
            family="command",
            candidate=300,
            baseline=100,
            ratio=3.0,
            source="hctracer",
        )
    )
    source_map = {
        "symbols": [
            {
                "kind": "graph_guard",
                "file": "engine/attention/graph.cpp",
            }
        ],
        "providers": [],
    }
    result = rank_paths(
        profile,
        wall_ratio=0.5,
        graph_timing=graph_timing(),
        graph_source=source_map,
    )

    assert result["top_path"] == "graph.source_guard"
    assert result["graph_evidence"]["paired_graph_timing"] is True
    assert result["ranked"][0]["source_files"] == ["engine/attention/graph.cpp"]
    assert any(item["path_id"] == "runtime.host_graph_command" for item in result["ranked"])


def test_explicit_graph_timing_selects_replay_without_source_map():
    result = rank_paths(
        graph_profile(),
        wall_ratio=0.5,
        graph_timing=graph_timing(),
    )

    assert result["top_path"] == "graph.replay_exclusive"
    assert result["graph_evidence"]["paired_graph_timing"] is True


def test_adjacent_graph_timing_does_not_claim_paired_replay():
    result = rank_paths(
        graph_profile(),
        wall_ratio=0.5,
        graph_timing=graph_timing(evidence_mode="EXCLUSIVE_UNION_WITH_ADJACENT_GUESS"),
    )

    assert result["graph_evidence"]["paired_graph_timing"] is False
    assert result["top_path"] == "diagnosis.insufficient_evidence"
