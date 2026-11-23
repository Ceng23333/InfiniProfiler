from infiniprofiler.engine_ranges import aggregate_engine_ranges, provenance_payload


def test_aggregate_engine_ranges_sums_named_spans():
    rows = [
        {"type": "span", "name": "engine.schedule", "duration_us": 10},
        {"type": "span", "name": "engine.schedule", "duration_us": 15},
        {"type": "metric", "name": "tokens_per_iteration", "value": 8},
        {"type": "span", "name": "engine.forward", "start_ns": 0, "end_ns": 2000},
    ]
    ranges, warnings = aggregate_engine_ranges(rows)
    by_name = {item["name"]: item for item in ranges}
    assert by_name["engine.schedule"]["count"] == 2
    assert by_name["engine.schedule"]["total_us"] == 25.0
    assert by_name["engine.forward"]["total_us"] == 2.0
    assert warnings == []
    payload = provenance_payload(ranges)
    assert payload["engine_ranges_contract"] == "provenance_only_not_scheduler_signals"
    assert len(payload["engine_ranges"]) == 2
