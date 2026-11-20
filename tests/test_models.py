from infiniprofiler.models import Capture, EngineMetric, ProfileDocument, merge_profiles


def test_merge_profiles_adds_fallback_sources_without_shadowing_capture():
    primary = ProfileDocument(capture=Capture(source="hctracer"))
    primary.metrics.append(EngineMetric(name="batch_size", candidate=8))
    fallback = ProfileDocument(capture=Capture(source="naive-sidecar"))
    fallback.metrics.append(EngineMetric(name="queue_depth", candidate=4))
    fallback.warnings.append("sidecar_only")

    merged = merge_profiles(primary, fallback)

    assert merged.capture.source == "hctracer"
    assert merged.capture.provenance["sources"] == ["naive-sidecar"]
    assert [metric.name for metric in merged.metrics] == ["batch_size", "queue_depth"]
    assert merged.warnings == ["sidecar_only"]
