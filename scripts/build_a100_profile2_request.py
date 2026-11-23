#!/usr/bin/env python3
"""Build profile-2 + AutoHelix request from matched A100 nsys baseline/candidate runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
	sys.path.insert(0, str(ROOT / "src"))

from infiniprofiler.capture import compare_capture_manifests
from infiniprofiler.contracts import validate_profile_v2
from infiniprofiler.models import (
	Capture,
	HostSignal,
	ProfileDocument,
	TimingAttributionFields,
	safe_ratio,
)
from infiniprofiler.promotion.autohelix import promotion_request
from infiniprofiler.promotion.ranker import rank_paths

ALLOWED_PATH_PREFIXES = ("runtime.", "graph.", "engine.", "diagnosis.insufficient_evidence")
TIMING_KEYS = (
	"api_submit_us",
	"gpu_execution_us",
	"launch_latency_us",
	"sync_wait_us",
	"graph_launch_us",
	"graph_replay_device_us",
	"cpu_gap_before_us",
	"cpu_gap_after_us",
)


def _load(path: Path) -> Dict[str, Any]:
	return json.loads(path.read_text(encoding="utf-8"))


def _dump(path: Path, payload: Mapping[str, Any]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sum_timing(merge: Mapping[str, Any]) -> Dict[str, Any]:
	totals = {key: 0.0 for key in TIMING_KEYS}
	unpaired = 0
	paired = 0
	modes: List[str] = []
	for row in merge.get("ranks") or []:
		timing = row.get("timing_attribution") or {}
		for key in TIMING_KEYS:
			totals[key] += float(timing.get(key) or 0.0)
		unpaired += int(timing.get("unpaired_event_count") or 0)
		paired += int(timing.get("paired_event_count") or 0)
		mode = timing.get("evidence_mode")
		if mode:
			modes.append(str(mode))
	evidence_mode = (
		"EXCLUSIVE_UNION_WITH_EXPLICIT_CORRELATION"
		if modes and all(m == "EXCLUSIVE_UNION_WITH_EXPLICIT_CORRELATION" for m in modes)
		else (modes[0] if modes else "UNKNOWN")
	)
	return {
		"evidence_mode": evidence_mode,
		**totals,
		"unpaired_event_count": unpaired,
		"paired_event_count": paired,
	}


def _host_signals(
	candidate: Mapping[str, Any],
	baseline: Mapping[str, Any],
) -> List[Dict[str, Any]]:
	signals: List[Dict[str, Any]] = []
	for family, key in (
		("api", "api_submit_us"),
		("runtime", "api_submit_us"),
		("graph", "graph_launch_us"),
		("gap", "cpu_gap_before_us"),
	):
		cand = float(candidate.get(key) or 0.0)
		base = float(baseline.get(key) or 0.0)
		if cand <= 0 and base <= 0:
			continue
		signals.append(
			{
				"family": family,
				"candidate": cand,
				"baseline": base,
				"ratio": safe_ratio(cand, base),
				"source": "nsys",
			}
		)
	return signals


def _wall_compare(baseline_client: Mapping[str, Any], candidate_client: Mapping[str, Any]) -> Dict[str, Any]:
	base_wall = float(baseline_client.get("wall_duration_ms") or 0.0)
	cand_wall = float(candidate_client.get("wall_duration_ms") or 0.0)
	base_tps = float(baseline_client.get("tokens_per_second") or 0.0)
	cand_tps = float(candidate_client.get("tokens_per_second") or 0.0)
	wall_ratio = safe_ratio(cand_wall, base_wall)
	return {
		"baseline": {
			"wall_duration_ms": base_wall,
			"tokens_per_second": base_tps,
			"completed_requests": baseline_client.get("completed_requests"),
			"failed_requests": baseline_client.get("failed_requests"),
		},
		"candidate": {
			"wall_duration_ms": cand_wall,
			"tokens_per_second": cand_tps,
			"completed_requests": candidate_client.get("completed_requests"),
			"failed_requests": candidate_client.get("failed_requests"),
		},
		"duration_ratio": wall_ratio or 0.0,
		"tokens_per_second_ratio": safe_ratio(cand_tps, base_tps),
		"wall_ratio": wall_ratio,
		"ratio": wall_ratio,
		"primary_wall_ratio": wall_ratio,
	}


def _assert_allowed_paths(ranked: Sequence[Mapping[str, Any]]) -> None:
	for item in ranked:
		path_id = str(item.get("path_id") or "")
		if not any(
			path_id == prefix or path_id.startswith(prefix)
			for prefix in ALLOWED_PATH_PREFIXES
		):
			raise SystemExit(f"disallowed ranked path_id={path_id!r}")


def build_profile_v2(
	*,
	baseline_dir: Path,
	candidate_dir: Path,
	engine_ranges_jsonl: Optional[Path] = None,
) -> Tuple[Dict[str, Any], ProfileDocument, Dict[str, Any], Dict[str, Any]]:
	baseline_merge = _load(baseline_dir / "profile/capture_merge.json")
	candidate_merge = _load(candidate_dir / "profile/capture_merge.json")
	baseline_manifest = baseline_dir / "profile/window_manifest.json"
	candidate_manifest = candidate_dir / "profile/window_manifest.json"
	candidate_manifest_raw = _load(candidate_manifest)
	rank_views = _load(candidate_dir / "profile/rank_views.json")
	baseline_client = _load(baseline_dir / "client/measured.json")
	candidate_client = _load(candidate_dir / "client/measured.json")
	capture_cmp = compare_capture_manifests(baseline_manifest, candidate_manifest)
	if capture_cmp.get("status") != "MATCHED":
		raise SystemExit(f"capture manifests not MATCHED: {capture_cmp.get('errors')}")

	identity = capture_cmp["identity"]
	baseline_timing = _sum_timing(baseline_merge)
	candidate_timing = _sum_timing(candidate_merge)
	host_signals = _host_signals(candidate_timing, baseline_timing)
	warnings: List[str] = []
	warnings.extend(str(w) for w in (candidate_merge.get("warnings") or []))
	warnings.extend(str(w) for w in (rank_views.get("warnings") or []))

	provenance: Dict[str, Any] = {
		"baseline_run": str(baseline_dir),
		"candidate_run": str(candidate_dir),
		"baseline_window_id": baseline_merge.get("window_id"),
		"candidate_window_id": candidate_merge.get("window_id"),
		"window_matched": True,
		"windows_matched": True,
		"baseline_timing_attribution": baseline_timing,
		"rank_views_status": rank_views.get("status") or candidate_merge.get("rank_views_status"),
		"capture_manifest_comparison": capture_cmp,
	}
	if engine_ranges_jsonl is not None:
		from infiniprofiler.tools.sidecar import normalize as normalize_sidecar

		sidecar = normalize_sidecar(engine_ranges_jsonl)
		ranges = (sidecar.capture.provenance or {}).get("engine_ranges") or []
		provenance.update(
			{
				"engine_ranges": ranges,
				"engine_ranges_contract": "provenance_only_not_scheduler_signals",
				"engine_ranges_input": str(engine_ranges_jsonl),
			}
		)
		warnings.extend(sidecar.warnings)
		warnings.append("engine_ranges_attached_provenance_only")

	started = int(candidate_manifest_raw["started_epoch_ns"])
	ended = int(candidate_manifest_raw["ended_epoch_ns"])
	start_skew_ms = 0.0
	if rank_views.get("rank_views"):
		start_skew_ms = max(
			float(row.get("start_skew_us") or 0.0) for row in rank_views["rank_views"]
		) / 1000.0

	profile_v2: Dict[str, Any] = {
		"schema_version": "infiniprofiler-profile-2",
		"capture": {
			"source": "nsys",
			"mode": "a100-matched-merge",
			"provenance": provenance,
		},
		"window": {
			"window_id": str(candidate_merge.get("window_id") or identity.get("wall_workload_id")),
			"wall_compare_ref": str(identity.get("wall_compare_ref") or "client/measured.json"),
			"wall_workload_id": str(identity.get("wall_workload_id")),
			"client_started_epoch_ns": started,
			"client_ended_epoch_ns": ended,
			"warmup_excluded": bool(identity.get("warmup_excluded")),
			"start_skew_ms": float(start_skew_ms),
			"duration_ratio": float(capture_cmp.get("duration_ratio") or 1.0),
			"alignment": "MATCHED",
		},
		"workload": {
			"campaign_id": str(identity.get("campaign_id")),
			"model": str(identity.get("model")),
			"backend_baseline_pin": str(identity.get("baseline_pin")),
			"backend_candidate_pin": str(identity.get("candidate_pin")),
			"tensor_parallel": int(identity.get("tensor_parallel")),
			"concurrency": int(identity.get("concurrency")),
			"prompt_lengths": list(identity.get("prompt_lengths") or []),
			"output_lengths": list(identity.get("output_lengths") or []),
			"batch_size": identity.get("batch_size"),
			"clock_policy": str(identity.get("clock_policy")),
			"profiles_source_hash": str(identity.get("profiles_source_hash")),
		},
		"ranks": list(rank_views.get("rank_views") or []),
		"timing_attribution": candidate_timing,
		"source_map": {
			"source_pin": str(identity.get("candidate_pin")),
			"source_root_hash": str(identity.get("profiles_source_hash")),
			"symbols": [],
			"providers": [],
			"scanned_file_count": 0,
			"artifacts": [
				{
					"kind": "nsys_merge",
					"path": str(candidate_dir / "profile/capture_merge.json"),
				}
			],
		},
		"host_signals": host_signals,
		"metrics": [],
		"warnings": warnings,
	}
	validate_profile_v2(profile_v2)

	doc = ProfileDocument(
		capture=Capture(
			source="nsys",
			mode="a100-matched-merge",
			provenance=provenance,
		),
		host_signals=[HostSignal(**item) for item in host_signals],
		timing_attribution=TimingAttributionFields.from_mapping(candidate_timing),
		metrics=[],
		warnings=warnings,
	)
	wall = _wall_compare(baseline_client, candidate_client)
	return profile_v2, doc, wall, rank_views


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--baseline", type=Path, required=True)
	parser.add_argument("--candidate", type=Path, required=True)
	parser.add_argument("--out-dir", type=Path, required=True)
	parser.add_argument(
		"--engine-ranges-jsonl",
		type=Path,
		default=None,
		help="optional L3 sidecar JSONL with span rows (provenance only)",
	)
	parser.add_argument("--host", default="A100")
	parser.add_argument("--device-track", default="nvidia-a100")
	parser.add_argument("--iteration", default="0")
	args = parser.parse_args()

	profile_v2, doc, wall, rank_views = build_profile_v2(
		baseline_dir=args.baseline.resolve(),
		candidate_dir=args.candidate.resolve(),
		engine_ranges_jsonl=args.engine_ranges_jsonl,
	)
	identity = profile_v2["workload"]
	request = promotion_request(
		doc,
		wall_compare=wall,
		campaign_id=str(identity["campaign_id"]),
		model=str(identity["model"]),
		host=args.host,
		device_track=args.device_track,
		baseline_pin=str(identity["backend_baseline_pin"]),
		candidate_pin=str(identity["backend_candidate_pin"]),
		iteration=args.iteration,
		graph_timing=profile_v2["timing_attribution"],
		graph_source=profile_v2["source_map"],
		graph_ranks=rank_views,
	)
	_assert_allowed_paths(request["diagnosis"]["ranked"])

	out_dir = args.out_dir.resolve()
	_dump(out_dir / "profile-v2.json", profile_v2)
	_dump(out_dir / "autohelix-request.json", request)
	_dump(
		out_dir / "timing_ratios.json",
		{
			"candidate": profile_v2["timing_attribution"],
			"baseline": profile_v2["capture"]["provenance"]["baseline_timing_attribution"],
			"wall_compare": wall,
		},
	)
	print(
		json.dumps(
			{
				"ok": True,
				"out_dir": str(out_dir),
				"top_path": request["diagnosis"]["top_path"],
				"stop_before_promote": request["stop_before_promote"],
				"ranked": [item["path_id"] for item in request["diagnosis"]["ranked"]],
			},
			indent=2,
			sort_keys=True,
		)
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
