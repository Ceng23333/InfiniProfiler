"""Semantic mcTracer analysis for matched-trace gates.

This adapter is intentionally numeric-first: it preserves the raw trace
categories and reports both summed busy time and interval-union coverage so
overlapping per-rank/device events are never double counted as wall time.
All durations are normalized to microseconds; mcTracer emits epoch-ns
timestamps and nanosecond durations, detected per event.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ..timeline import _event_args, _is_device, _is_host, stream_chrome_events

# Nsight-pattern two-layer taxonomy generalized to any *PTI tracer
# (CUPTI, MCPTI/MACA mcTracer, ROCTracer/rocprofiler, ...).
# Layer 1: activity kinds, the device-agnostic equivalent of CUPTI tables.
ACTIVITY_KINDS = ("kernel", "memcpy", "memset", "sync", "host_api", "graph")

# Layer 2: semantic buckets, applied only to device kernels by name.
SEMANTIC_BUCKETS = (
    "attention",
    "gemm",
    "allreduce",
    "residual_rmsnorm",
    "mlp_activation",
    "rope",
    "embedding",
    "kv_cache_reshape",
    "sampling",
    "graph",
    "other",
)


def bucket_keys():
    """Stable gate family keys: activity kinds only (Layer 1 contract).

    Semantic classification is a non-contract helper for ad-hoc inspection;
    gate output never depends on kernel-name heuristics.
    """
    return ACTIVITY_KINDS


# Epoch-ns timestamps from mcTracer are ~1.8e18; microseconds stay < 1e15.
NS_TIMESTAMP_THRESHOLD = 1e15

ATTENTION_RE = __import__("re").compile(r"flash|attn|attention|softmax|sdpa", __import__("re").I)
GEMM_RE = __import__("re").compile(r"gemm|gemv|matmul|blas|sgemm|hgemm|bgemm", __import__("re").I)
ALLREDUCE_RE = __import__("re").compile(r"allreduce|all_reduce|nccl|rccl|mccl", __import__("re").I)
NORM_RE = __import__("re").compile(r"rmsnorm|layernorm|layer_norm|\bnorm", __import__("re").I)
ROPE_RE = __import__("re").compile(r"rope|rotary", __import__("re").I)
KV_RE = __import__("re").compile(r"paged|kv[_ ]?cache|reshape|transpose|cache", __import__("re").I)
MEMCPY_RE = __import__("re").compile(r"memcpy|htod|dtoh|dtod|copy", __import__("re").I)
MEMSET_RE = __import__("re").compile(r"memset|fill", __import__("re").I)
MLP_RE = __import__("re").compile(r"swiglu|silu|gelu|mlp|ffn|feedforward", __import__("re").I)
EMBED_RE = __import__("re").compile(r"embedding|embed", __import__("re").I)
REARRANGE_RE = __import__("re").compile(r"rearrange|permute|permute_2d|contiguous", __import__("re").I)
# Triton fused kernels carry op hints instead of module names (vLLM).
TRITON_NORM_RE = __import__("re").compile(r"fused_add_mul_sub|fused_rms", __import__("re").I)
TRITON_MLP_RE = __import__("re").compile(r"fused_mul_silu|fused_mlp", __import__("re").I)
SAMPLING_RE = __import__("re").compile(r"sample|sampling|argmax|topk|topp|softmax_topk", __import__("re").I)
GRAPH_RE = __import__("re").compile(r"graph|capture|replay", __import__("re").I)
SCHED_RE = __import__("re").compile(
    r"scheduler|schedule|batch|step|forward|admission|queue", __import__("re").I
)


def classify_semantic(name: str, cat: Optional[str]) -> Tuple[str, str]:
    """Return ``(activity_kind, semantic)`` for one device-side event.

    ``semantic`` is only meaningful for ``kernel``; memcpy/memset are their own
    activity kinds, matching the CUPTI/nsys table split.
    """
    n = str(name or "")
    lowered = n.lower()
    category = str(cat or "").lower()
    if category == "memset" or MEMSET_RE.search(lowered):
        return "memset", ""
    if category == "memcpy" or MEMCPY_RE.search(lowered):
        return "memcpy", ""
    if TRITON_NORM_RE.search(lowered):
        return "kernel", "residual_rmsnorm"
    if TRITON_MLP_RE.search(lowered):
        return "kernel", "mlp_activation"
    if ATTENTION_RE.search(lowered):
        return "kernel", "attention"
    if GEMM_RE.search(lowered):
        return "kernel", "gemm"
    if ALLREDUCE_RE.search(lowered):
        return "kernel", "allreduce"
    if "rmsnorm" in lowered or "layernorm" in lowered or "layer_norm" in lowered:
        return "kernel", "residual_rmsnorm"
    if ROPE_RE.search(lowered):
        return "kernel", "rope"
    if MLP_RE.search(lowered):
        return "kernel", "mlp_activation"
    if EMBED_RE.search(lowered):
        return "kernel", "embedding"
    if SAMPLING_RE.search(lowered):
        return "kernel", "sampling"
    if GRAPH_RE.search(lowered):
        return "kernel", "graph"
    if KV_RE.search(lowered) or REARRANGE_RE.search(lowered):
        return "kernel", "kv_cache_reshape"
    if SCHED_RE.search(lowered):
        return "kernel", "other"
    if NORM_RE.search(lowered):
        return "kernel", "residual_rmsnorm"
    return "kernel", "other"


def _unit_scale(event: Mapping[str, Any]) -> float:
    ts = event.get("ts")
    try:
        if ts is not None and float(ts) >= NS_TIMESTAMP_THRESHOLD:
            return 0.001  # ns -> us
    except (TypeError, ValueError):
        pass
    return 1.0


def _is_device_kernelish(event, name: str) -> bool:
    """Device-side fallback for kernel symbols lacking device args/cats."""
    if _is_device(event, name):
        return True
    if _is_host(event, name):
        return False
    lowered = name.lower()
    return (
        "<" in name
        or "::" in name
        or name.startswith("_Z")
        or name.startswith("triton_")
        or lowered.endswith("_kernel")
        or "kernel" in lowered
    )


def _union(intervals: List[Tuple[float, float]]) -> float:
    total = 0.0
    active_start: Optional[float] = None
    active_end = 0.0
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if active_start is None:
            active_start, active_end = start, end
        elif start <= active_end:
            active_end = max(active_end, end)
        else:
            total += active_end - active_start
            active_start, active_end = start, end
    if active_start is not None:
        total += active_end - active_start
    return max(0.0, total)


def analyze_trace(path: Path, *, client_window_s: Optional[float] = None) -> Dict[str, Any]:
    warnings: List[str] = []
    bucket_intervals: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    bucket_streams: Dict[Tuple[str, int, int, str], List[Tuple[float, float]]] = defaultdict(list)
    bucket_counts: Dict[str, int] = defaultdict(int)
    bucket_summed: Dict[str, float] = defaultdict(float)
    host_intervals: List[Tuple[float, float]] = []
    raw_cat_counts: Dict[str, int] = defaultdict(int)
    raw_cat_ns: Dict[str, float] = defaultdict(float)
    stream_device_intervals: Dict[Tuple[int, int, str], List[Tuple[float, float]]] = defaultdict(list)
    # Float64 cannot hold us durations added to epoch-ns timestamps (~1.7e15,
    # ulp 0.25us). Store all intervals relative to the first observed start.
    time_base: Optional[float] = None

    for event, new_warnings in stream_chrome_events(path):
        for warning in new_warnings:
            if warning not in warnings:
                warnings.append(warning)
        if event is None:
            continue
        if event.get("ph") != "X":
            if event.get("cat") is not None:
                raw_cat_counts[str(event.get("cat"))] += 1
            continue
        name = str(event.get("name") or "")
        try:
            dur = float(event.get("dur", 0.0))
            start = float(event.get("ts", 0.0))
        except (TypeError, ValueError):
            continue
        if dur <= 0:
            continue
        scale = _unit_scale(event)
        dur_us = dur * scale
        start_us = start * scale
        cat = event.get("cat")
        cat_key = str(cat) if cat is not None else "null"
        raw_cat_counts[cat_key] += 1
        raw_cat_ns[cat_key] += dur
        args = _event_args(event)
        try:
            pid = int(event.get("pid") or 0)
            tid = int(event.get("tid") or 0)
        except (TypeError, ValueError):
            pid, tid = 0, 0
        stream = str(args.get("stream_id") or args.get("streamId") or args.get("stream") or tid)
        if time_base is None:
            time_base = start_us
        rel_start = start_us - time_base
        if _is_device_kernelish(event, name):
            kind, _semantic = classify_semantic(name, cat if isinstance(cat, str) else None)
            bucket = kind
            bucket_intervals[bucket].append((rel_start, rel_start + dur_us))
            bucket_counts[bucket] += 1
            bucket_summed[bucket] += dur_us
            key = (pid, tid, stream)
            stream_device_intervals[key].append((rel_start, rel_start + dur_us))
            bucket_streams[(bucket, pid, tid, stream)].append((rel_start, rel_start + dur_us))
        elif _is_host(event, name):
            host_intervals.append((rel_start, rel_start + dur_us))

    device_intervals: List[Tuple[float, float]] = []
    for intervals in stream_device_intervals.values():
        device_intervals.extend(intervals)

    def _bounds(intervals: List[Tuple[float, float]]) -> Tuple[Optional[float], Optional[float]]:
        if not intervals:
            return None, None
        return min(i[0] for i in intervals), max(i[1] for i in intervals)

    dev_start, dev_end = _bounds(device_intervals)
    host_start, host_end = _bounds(host_intervals)
    span_start = min(x for x in (dev_start, host_start) if x is not None)
    span_end = max(x for x in (dev_end, host_end) if x is not None)
    window_us = (span_end - span_start) if span_end > span_start else 0.0

    host_coverage = _union(host_intervals)
    device_coverage = _union(device_intervals)
    buckets: List[Dict[str, Any]] = []
    for bucket in bucket_keys():
        intervals = bucket_intervals.get(bucket) or []
        if not intervals:
            continue
        coverage = _union(intervals)
        buckets.append(
            {
                "bucket": bucket,
                "activity_kind": bucket,
                "count": bucket_counts[bucket],
                "summed_us": round(bucket_summed[bucket], 3),
                "coverage_us": round(coverage, 3),
                "coverage_share": round(coverage / window_us, 6) if window_us else None,
            }
        )

    streams: List[Dict[str, Any]] = []
    for (pid, tid, stream), intervals in sorted(stream_device_intervals.items()):
        coverage = _union(intervals)
        streams.append(
            {
                "pid": pid,
                "tid": tid,
                "stream": stream,
                "event_count": len(intervals),
                "coverage_us": round(coverage, 3),
                "coverage_share_of_window": (
                    round(coverage / window_us, 6) if window_us else None
                ),
            }
        )

    return {
        "schema_version": "mctracer-activity-1",
        "source": str(path),
        "time_unit_detected": "ns",
        "warnings": warnings,
        "trace_span_us": round(window_us, 3),
        "client_window_s": client_window_s,
        "buckets": buckets,
        "streams": streams,
        "host_submit": {
            "coverage_us": round(host_coverage, 3),
            "idle_us": round(max(0.0, window_us - host_coverage), 3),
            "idle_share": (
                round(max(0.0, window_us - host_coverage) / window_us, 6) if window_us else None
            ),
        },
        "device": {
            "coverage_us": round(device_coverage, 3),
            "idle_us": round(max(0.0, window_us - device_coverage), 3),
            "idle_share": (
                round(max(0.0, window_us - device_coverage) / window_us, 6) if window_us else None
            ),
        },
        "raw_categories": {
            "counts": dict(sorted(raw_cat_counts.items())),
            "durations_ns": {k: round(v, 3) for k, v in sorted(raw_cat_ns.items())},
        },
        "window_notes": [
            "window_us is the trace span; per-client-wall rates need client_window_s",
        ],
    }


def compare_gate(
    candidate: Dict[str, Any],
    baseline: Dict[str, Any],
    *,
    candidate_window_s: Optional[float] = None,
    baseline_window_s: Optional[float] = None,
) -> Dict[str, Any]:
    cand_by = {b["bucket"]: b for b in candidate["buckets"]}
    base_by = {b["bucket"]: b for b in baseline["buckets"]}
    cand_total = float(candidate["device"]["coverage_us"]) or 1.0
    base_total = float(baseline["device"]["coverage_us"]) or 1.0
    cand_window_s = candidate_window_s or candidate.get("client_window_s")
    base_window_s = baseline_window_s or baseline.get("client_window_s")
    families: Dict[str, Any] = {}
    for bucket in bucket_keys():
        cand = float(cand_by.get(bucket, {}).get("coverage_us", 0.0))
        base = float(base_by.get(bucket, {}).get("coverage_us", 0.0))
        if cand <= 0 and base <= 0:
            continue
        cand_share = cand / cand_total
        base_share = base / base_total
        families[bucket] = {
            "activity_kind": bucket,
            "candidate_us": round(cand, 3),
            "baseline_us": round(base, 3),
            "ratio": round(cand / base, 6) if base > 0 else None,
            "candidate_share": round(cand_share, 6),
            "baseline_share": round(base_share, 6),
            "share_gap_pp": round((cand_share - base_share) * 100, 4),
            "magnitude_gap_us": round(cand - base, 3),
        }
        if cand_window_s:
            families[bucket]["candidate_us_per_client_s"] = round(cand / cand_window_s, 3)
        if base_window_s:
            families[bucket]["baseline_us_per_client_s"] = round(base / base_window_s, 3)
    overall = {
        "candidate_total_us": round(float(candidate["device"]["coverage_us"]), 3),
        "baseline_total_us": round(float(baseline["device"]["coverage_us"]), 3),
        "gap_magnitude_us": round(
            float(candidate["device"]["coverage_us"]) - float(baseline["device"]["coverage_us"]), 3
        ),
        "gap_ratio": round(
            float(candidate["device"]["coverage_us"]) / float(baseline["device"]["coverage_us"]), 6
        )
        if float(baseline["device"]["coverage_us"]) > 0
        else None,
        "candidate_device_idle_us": candidate["device"]["idle_us"],
        "baseline_device_idle_us": baseline["device"]["idle_us"],
        "candidate_host_submit_idle_us": candidate["host_submit"]["idle_us"],
        "baseline_host_submit_idle_us": baseline["host_submit"]["idle_us"],
    }
    if candidate_window_s:
        overall["candidate_us_per_client_s"] = round(
            float(candidate["device"]["coverage_us"]) / candidate_window_s, 3
        )
    if baseline_window_s:
        overall["baseline_us_per_client_s"] = round(
            float(baseline["device"]["coverage_us"]) / baseline_window_s, 3
        )
    return {
        "schema_version": "mctracer-activity-gate-1",
        "families": families,
        "overall": overall,
    }


def load_client_window(path: Optional[Path]) -> Optional[float]:
    if path is None:
        return None
    doc = json.loads(path.read_text(encoding="utf-8"))
    elapsed = doc.get("elapsed_s")
    return float(elapsed) if elapsed else None
