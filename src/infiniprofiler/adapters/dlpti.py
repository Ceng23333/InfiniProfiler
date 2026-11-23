"""Native DLPTI export normalizer (diverge path; not Chrome traceEvents).

``dlpti_tools export --format json`` emits ``activities[]`` with numeric
``kind`` values from ``DLPTI_ActivityKind``. Timed records carry ``start`` /
``end`` timestamps in nanoseconds. ``perfetto-json`` is not used: on SDK
0.10.0 it emits broken nested ``traceEvents`` wrappers.

Wrap-grade path streams multi-GiB exports and emits CU bucket metrics.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from ..models import Capture, EngineMetric, HostSignal, ProfileDocument, safe_ratio
from .dlpti_buckets import (
	COMPARE_BUCKETS,
	MATERIAL_BUCKETS,
	METRIC_PREFIX,
	bucket_for,
	empty_bucket_map,
	metric_name,
	select_bucket_stat,
)

# Re-export for tests / callers that inspect material set.
__all__ = ["normalize", "iter_activities", "load_native_export", "detect_smoke_fingerprint"]


# From /usr/local/dlgpu/sdk/include/dlpti/activity.h
KIND_DRIVER_API = 1
KIND_RUNTIME_API = 2
KIND_CU_KERNEL = 20
KIND_TU_KERNEL = 21
KIND_GRAPH = 32
KIND_GRAPH_INSTANCE = 33
KIND_GRAPH_ADD_NODE = 34
KIND_GRAPH_V1 = 35
KIND_COMMAND = 49
KIND_MEMCPY = 50
KIND_MEMCPY4D = 51
KIND_MEMCPY6D = 52
KIND_MEMSET = 53
KIND_MEMSET4D = 54
KIND_MEMSET6D = 55
KIND_CU = 56
KIND_FLUSH = 57
KIND_TU = 58
KIND_MARKER = 82

LAUNCH_RE = re.compile(r"launch", re.IGNORECASE)
GRAPH_NAME_RE = re.compile(r"graph", re.IGNORECASE)
SMOKE_CMDLINE_RE = re.compile(r"vector[_-]?add", re.IGNORECASE)

# Activity-count floor below which a wrap export is treated as smoke / empty.
SMOKE_ACTIVITY_FLOOR = 200
# Full-load path only for tiny fixtures (bytes).
FULL_LOAD_MAX_BYTES = 8 * 1024 * 1024

GRAPH_KINDS = {
	KIND_GRAPH,
	KIND_GRAPH_INSTANCE,
	KIND_GRAPH_ADD_NODE,
	KIND_GRAPH_V1,
}
MEMCPY_KINDS = {
	KIND_MEMCPY,
	KIND_MEMCPY4D,
	KIND_MEMCPY6D,
	KIND_MEMSET,
	KIND_MEMSET4D,
	KIND_MEMSET6D,
}
KERNEL_KINDS = {KIND_CU_KERNEL, KIND_TU_KERNEL, KIND_CU, KIND_TU}


def iter_activities(path: Path) -> Iterator[Dict[str, Any]]:
	"""Yield activity objects without loading the whole file.

	The exporter writes pretty-printed JSON with one activity object per line
	inside the ``"activities"`` array. Full ``json.load`` is unsafe for GiB traces.
	"""
	inside = False
	with path.open(encoding="utf-8", errors="replace") as stream:
		for line in stream:
			if not inside:
				if '"activities"' in line and "[" in line:
					inside = True
				continue
			text = line.strip()
			if text.startswith("]"):
				return
			if text.endswith(","):
				text = text[:-1]
			if not text:
				continue
			obj = json.loads(text)
			if isinstance(obj, dict):
				yield obj


def _read_header_fields(path: Path, max_bytes: int = 512_000) -> Dict[str, Any]:
	"""Best-effort extract of profiler_info / profile_targets from file head."""
	with path.open("rb") as handle:
		raw = handle.read(max_bytes)
	text = raw.decode("utf-8", errors="replace")
	out: Dict[str, Any] = {}
	m = re.search(
		r'"profiler_info"\s*:\s*(\[.*?\])\s*,\s*"profile_targets"', text, re.S
	)
	if m:
		try:
			out["profiler_info"] = json.loads(m.group(1))
		except json.JSONDecodeError:
			pass
	m2 = re.search(
		r'"profile_targets"\s*:\s*(\[.*?\])\s*,\s*"activities"', text, re.S
	)
	if m2:
		try:
			out["profile_targets"] = json.loads(m2.group(1))
		except json.JSONDecodeError:
			pass
	return out


def load_native_export(path: Path) -> Tuple[Dict[str, Any], List[str]]:
	"""Load a native DLPTI JSON export with an ``activities`` array.

	Tiny fixtures use full ``json.load``; larger files stream activities.
	"""
	warnings: List[str] = []
	size = path.stat().st_size
	if size <= FULL_LOAD_MAX_BYTES:
		text = path.read_text(encoding="utf-8", errors="replace")
		doc = json.loads(text)
		if not isinstance(doc, dict) or not isinstance(doc.get("activities"), list):
			raise ValueError("root is not a DLPTI native export with activities[]")
		return doc, warnings

	head = path.read_bytes()[:4096].decode("utf-8", errors="replace")
	if '"activities"' not in head:
		raise ValueError("root is not a DLPTI native export with activities[]")
	header = _read_header_fields(path)
	activities = list(iter_activities(path))
	doc = {
		"profiler_info": header.get("profiler_info") or [],
		"profile_targets": header.get("profile_targets") or [],
		"activities": activities,
	}
	warnings.append(
		f"dlpti_streamed_parse: size_bytes={size} activities={len(activities)}"
	)
	return doc, warnings


def _activity_name(act: Dict[str, Any]) -> str:
	for key in ("name", "api_name", "kernel_name", "node_name"):
		value = act.get(key)
		if value not in (None, "", "null"):
			return str(value)
	return f"kind:{act.get('kind')}"


def _duration_us(act: Dict[str, Any]) -> Optional[float]:
	"""Return activity duration in microseconds when start/end are present."""
	for start_key, end_key, scale in (
		("start", "end", 1e-3),  # ns → us
		("start_time", "end_time", 1.0),
		("start_us", "end_us", 1.0),
	):
		if start_key not in act or end_key not in act:
			continue
		try:
			start = float(act[start_key])
			end = float(act[end_key])
		except (TypeError, ValueError):
			continue
		if end < start:
			continue
		delta = end - start
		# Heuristic: values that look like absolute ns timestamps → us
		if scale == 1e-3 and delta > 1e6:
			return delta * scale
		if scale == 1e-3 and start > 1e15:
			return delta * scale
		if "duration" in act:
			try:
				return max(0.0, float(act["duration"]))
			except (TypeError, ValueError):
				pass
		return max(0.0, delta * scale)
	for key in ("duration", "dur", "elapsed_us", "time_us"):
		if key not in act:
			continue
		try:
			val = float(act[key])
		except (TypeError, ValueError):
			continue
		if key.endswith("_us") or (key == "duration" and val > 1000):
			return max(0.0, val if key.endswith("_us") else val / 1000.0)
		return max(0.0, val)
	return None


def _cmdline_blobs(doc: Dict[str, Any]) -> List[str]:
	blobs: List[str] = []
	for section in ("profiler_info", "profile_targets"):
		for entry in doc.get(section) or []:
			if not isinstance(entry, dict):
				continue
			info = entry.get("process_info") or {}
			if isinstance(info, dict) and info.get("cmdline"):
				blobs.append(str(info["cmdline"]))
	return blobs


def detect_smoke_fingerprint(
	doc: Dict[str, Any],
	*,
	activity_count: int,
	kernel_count: int,
	material_kernel_us: float = 0.0,
) -> Optional[str]:
	"""Return a reason string when the export looks like vectorAdd / empty smoke.

	Tiny synthetic unit fixtures with real LLM op names are not treated as smoke
	solely for low activity_count; the activity floor applies when there is no
	material CU mass (or cmdline is vector_add).
	"""
	cmdlines = _cmdline_blobs(doc)
	joined = " | ".join(cmdlines)
	if any(SMOKE_CMDLINE_RE.search(c) for c in cmdlines):
		return f"smoke_fingerprint: vector_add cmdline in provenance ({joined[:160]})"
	if activity_count < SMOKE_ACTIVITY_FLOOR and material_kernel_us <= 0:
		return (
			f"smoke_fingerprint: activity_count={activity_count} "
			f"< floor={SMOKE_ACTIVITY_FLOOR}"
		)
	# Tiny SDK sample: only a handful of CU kernels and no LLM op mass.
	if (
		kernel_count > 0
		and kernel_count <= 8
		and activity_count < 100
		and material_kernel_us <= 0
	):
		return (
			f"smoke_fingerprint: only_sdk_sample_kinds "
			f"kernel_count={kernel_count} activity_count={activity_count}"
		)
	return None


def extract_host_totals_us(
	doc: Dict[str, Any],
) -> Tuple[Dict[str, float], Dict[str, Any], Dict[str, Dict[str, float]]]:
	totals = {
		"api": 0.0,
		"command": 0.0,
		"driver": 0.0,
		"runtime": 0.0,
		"kernel": 0.0,
		"memcpy": 0.0,
		"graph": 0.0,
		"gap": 0.0,
	}
	counts = {key: 0 for key in totals}
	bucket_sums = empty_bucket_map()
	timed = 0
	untimed = 0
	for raw in doc.get("activities") or []:
		if not isinstance(raw, dict):
			continue
		try:
			kind = int(raw.get("kind"))
		except (TypeError, ValueError):
			continue
		name = _activity_name(raw)
		dur = _duration_us(raw)
		if dur is None:
			untimed += 1
			# Count proxy (1 us per record) so empty-duration structural
			# records still contribute comparable family mass.
			dur = 1.0
			proxy = True
		else:
			timed += 1
			proxy = False

		if kind == KIND_DRIVER_API:
			totals["driver"] += dur
			totals["api"] += dur
			counts["driver"] += 1
			counts["api"] += 1
			if LAUNCH_RE.search(name):
				totals["command"] += dur
				counts["command"] += 1
		elif kind == KIND_RUNTIME_API:
			totals["runtime"] += dur
			totals["api"] += dur
			counts["runtime"] += 1
			counts["api"] += 1
			if LAUNCH_RE.search(name):
				totals["command"] += dur
				counts["command"] += 1
		elif kind == KIND_COMMAND:
			totals["command"] += dur
			counts["command"] += 1
		elif kind in GRAPH_KINDS or GRAPH_NAME_RE.search(name):
			totals["graph"] += dur
			counts["graph"] += 1
		elif kind in MEMCPY_KINDS or "memcpy" in name.lower() or "memset" in name.lower():
			totals["memcpy"] += dur
			counts["memcpy"] += 1
		elif kind in KERNEL_KINDS:
			totals["kernel"] += dur
			counts["kernel"] += 1
			b = bucket_for(name)
			bucket_sums.setdefault(b, {"sum_us": 0.0, "count": 0.0})
			bucket_sums[b]["sum_us"] += dur
			bucket_sums[b]["count"] += 1.0
		elif kind == KIND_FLUSH:
			# Flush is device sync overhead; fold into gap numerator later.
			totals["gap"] += dur
			counts["gap"] += 1
		elif proxy:
			# Ignore untimed metadata (device/context/stream) for host families.
			pass

	busy = totals["kernel"] + totals["memcpy"]
	submitted = totals["api"] + totals["command"]
	computed_gap = max(0.0, submitted - busy) if submitted > 0 else 0.0
	totals["gap"] = max(totals["gap"], computed_gap)
	meta = {
		"timed_activities": timed,
		"untimed_activities": untimed,
		"counts": counts,
		"activity_count": timed + untimed,
		"kernel_count": counts["kernel"],
	}
	return totals, meta, bucket_sums


def _host_signals(
	totals: Dict[str, float],
	baseline: Optional[Dict[str, float]],
) -> List[HostSignal]:
	output: List[HostSignal] = []
	# Wrap traces move kernel/memcpy mass; include them alongside host families.
	for family in ("driver", "runtime", "command", "api", "gap", "graph", "kernel", "memcpy"):
		candidate = float(totals.get(family, 0.0))
		base = float(baseline.get(family, 0.0)) if baseline else None
		if candidate <= 0 and (base is None or base <= 0):
			continue
		ratio = safe_ratio(candidate, base) if base else None
		output.append(
			HostSignal(
				family=family,
				candidate=candidate,
				baseline=base,
				ratio=ratio,
				source="dlpti",
			)
		)
	return output


def _cu_bucket_metrics(
	bucket_sums: Dict[str, Dict[str, float]],
	baseline_sums: Optional[Dict[str, Dict[str, float]]],
	*,
	emit: bool,
) -> List[EngineMetric]:
	if not emit:
		return []
	metrics: List[EngineMetric] = []
	for name in COMPARE_BUCKETS:
		stat = select_bucket_stat(bucket_sums, name)
		cand = float(stat.get("sum_us") or 0.0)
		base = None
		if baseline_sums is not None:
			base_stat = select_bucket_stat(baseline_sums, name)
			base = float(base_stat.get("sum_us") or 0.0)
		if cand <= 0 and (base is None or base <= 0):
			continue
		ratio = safe_ratio(cand, base) if base is not None and base > 0 else None
		metrics.append(
			EngineMetric(
				name=metric_name(name),
				candidate=cand,
				baseline=base,
				ratio=ratio,
				unit="us",
				source="dlpti",
			)
		)
	return metrics


def normalize(
	input_path: Path,
	*,
	source_name: str = "dlpti",
	mode: str = "wrap",
	pid: Optional[int] = None,
	baseline_path: Optional[Path] = None,
) -> ProfileDocument:
	doc, warnings = load_native_export(input_path)
	totals, meta, bucket_sums = extract_host_totals_us(doc)
	baseline_totals = None
	baseline_buckets = None
	if baseline_path is not None:
		baseline_doc, baseline_warnings = load_native_export(baseline_path)
		baseline_totals, baseline_meta, baseline_buckets = extract_host_totals_us(
			baseline_doc
		)
		warnings.extend("baseline:" + item for item in baseline_warnings)
		meta["baseline"] = baseline_meta

	if meta["timed_activities"] == 0 and meta["activity_count"] > 0:
		warnings.append(
			"dlpti_no_duration_fields: activities lack start/end timing; "
			"host signals use 1us count proxies and are diagnostic only"
		)
	warnings.append(
		"no_scheduler_annotations: DLPTI native export has no InfiniLM "
		"HCTX/marker ranges; host gaps are not scheduler evidence"
	)

	material_us = 0.0
	for name in MATERIAL_BUCKETS:
		material_us += float(select_bucket_stat(bucket_sums, name).get("sum_us") or 0.0)
	smoke = detect_smoke_fingerprint(
		doc,
		activity_count=int(meta["activity_count"]),
		kernel_count=int(meta["kernel_count"]),
		material_kernel_us=material_us,
	)
	non_smoke_gate = os.environ.get("NON_SMOKE_GATE", "").strip() in ("1", "true", "yes")
	emit_cu = True
	if smoke:
		warnings.append(smoke)
		warnings.append(
			"dlpti_smoke_guard: refusing promotion-grade CU bucket metrics "
			"(empty metrics so ranker cannot treat smoke as serve evidence)"
		)
		emit_cu = False
		if non_smoke_gate:
			warnings.append("NON_SMOKE_GATE=1: smoke fingerprint hard-fail marker set")

	capture = Capture(
		source=source_name,
		mode=mode,
		pid=pid,
		provenance={
			"input": str(input_path),
			"baseline_input": str(baseline_path) if baseline_path else None,
			"activity_count": meta["activity_count"],
			"timed_activities": meta["timed_activities"],
			"kernel_count": meta["kernel_count"],
			"vendor_export": "dlpti_tools --format json",
			"smoke_fingerprint": smoke,
			"cu_bucket_prefix": METRIC_PREFIX,
			"material_buckets": sorted(MATERIAL_BUCKETS),
		},
	)
	return ProfileDocument(
		capture=capture,
		host_signals=_host_signals(totals, baseline_totals),
		scheduler_signals=[],
		metrics=_cu_bucket_metrics(bucket_sums, baseline_buckets, emit=emit_cu),
		warnings=warnings,
	)
