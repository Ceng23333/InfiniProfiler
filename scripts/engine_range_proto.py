#!/usr/bin/env python3
"""Prototype optional L3 engine-range emit + provenance ingest (no scheduler_signals)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
	sys.path.insert(0, str(ROOT / "src"))

from infiniprofiler.engine_ranges import aggregate_engine_ranges, provenance_payload
from infiniprofiler.tools.sidecar import normalize
from infiniprofiler.tools.sidecar_producer import SidecarSpanRecorder


# Named ranges that engines may emit. These are NOT InfiniLM canonical contracts
# and must never be reconstructed from CUPTI idle gaps.
PROTO_RANGES = (
	"engine.schedule",
	"engine.admission",
	"engine.batch_build",
	"engine.forward",
)


def emit_demo_jsonl(path: Path, *, window_id: str, rank: int = 0) -> None:
	"""Emit a short host-loop demo with metrics + optional ranges."""
	path.parent.mkdir(parents=True, exist_ok=True)
	if path.exists():
		path.unlink()
	recorder = SidecarSpanRecorder(path, rank=rank, window_id=window_id)
	base_ns = time.time_ns()
	# Simulate a few schedule→admit→batch→forward iterations with free-form names.
	cursor = base_ns
	for iteration in range(1, 4):
		for name, duration_us in (
			(PROTO_RANGES[0], 120.0 + iteration),
			(PROTO_RANGES[1], 40.0),
			(PROTO_RANGES[2], 80.0 + 2 * iteration),
			(PROTO_RANGES[3], 900.0 + 10 * iteration),
		):
			start = cursor
			end = start + int(duration_us * 1000)
			# Prefer structured duration fields for aggregate_engine_ranges.
			with path.open("a", encoding="utf-8") as handle:
				handle.write(
					json.dumps(
						{
							"type": "span",
							"name": name,
							"window_id": window_id,
							"rank": rank,
							"iteration": iteration,
							"start_ns": start,
							"end_ns": end,
							"duration_us": duration_us,
						},
						sort_keys=True,
					)
					+ "\n"
				)
			cursor = end + 1_000
		recorder.metric("tokens_per_iteration", 16 + iteration, iteration=iteration)
		recorder.metric("queue_depth", 2, iteration=iteration)


def ingest(path: Path) -> dict:
	profile = normalize(path)
	payload = provenance_payload(
		(profile.capture.provenance or {}).get("engine_ranges") or []
	)
	# Prove ranker does not invent scheduler.* from these ranges alone.
	from infiniprofiler.promotion.ranker import rank_paths

	backlog = rank_paths(profile, wall_ratio=1.0)
	for item in backlog["ranked"]:
		path_id = str(item["path_id"])
		if path_id.startswith("scheduler."):
			raise SystemExit(f"engine-range prototype invented forbidden path {path_id}")
	return {
		"input": str(path),
		"warnings": profile.warnings,
		"provenance": {
			**(profile.capture.provenance or {}),
			**payload,
		},
		"metrics": [m.to_dict() for m in profile.metrics],
		"ranked_paths": [item["path_id"] for item in backlog["ranked"]],
		"top_path": backlog["top_path"],
		"note": (
			"L3 ranges are provenance-only until matched A/B window identity and "
			"ratio thresholds exist; never map CUPTI cpu_gap_* to packing."
		),
	}


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--out-jsonl", type=Path, required=True)
	parser.add_argument("--out-report", type=Path, required=True)
	parser.add_argument("--window-id", default="engine-range-proto-window")
	parser.add_argument("--rank", type=int, default=0)
	parser.add_argument(
		"--emit-only",
		action="store_true",
		help="only write the demo JSONL (skip normalize/rank assert)",
	)
	args = parser.parse_args()

	emit_demo_jsonl(args.out_jsonl, window_id=args.window_id, rank=args.rank)
	if args.emit_only:
		print(json.dumps({"ok": True, "jsonl": str(args.out_jsonl)}, indent=2))
		return 0

	report = ingest(args.out_jsonl)
	args.out_report.parent.mkdir(parents=True, exist_ok=True)
	args.out_report.write_text(
		json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
	)
	# Also surface a compact aggregate for docs/demos.
	rows = []
	for line in args.out_jsonl.read_text(encoding="utf-8").splitlines():
		if line.strip():
			rows.append(json.loads(line))
	ranges, _ = aggregate_engine_ranges(rows)
	print(
		json.dumps(
			{
				"ok": True,
				"jsonl": str(args.out_jsonl),
				"report": str(args.out_report),
				"engine_ranges": ranges,
				"top_path": report["top_path"],
				"ranked_paths": report["ranked_paths"],
			},
			indent=2,
			sort_keys=True,
		)
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
