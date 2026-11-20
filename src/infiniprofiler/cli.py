"""Command-line interface for normalization and promotion bridging."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from .adapters.hctracer import normalize as normalize_hctracer
from .models import ProfileDocument
from .promotion.autohelix import promotion_request, write_io_compat
from .promotion.ranker import primary_wall_ratio, rank_paths
from .tools.sidecar import normalize as normalize_sidecar
from .tools.process_sampler import discover_pids, sample_interval, write_jsonl


def _load_json(path: Path) -> dict:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return doc


def _add_identity_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--model", default="Qwen3-8B")
    parser.add_argument("--host", default="metax-9-sg")
    parser.add_argument("--device-track", default="mars", choices=["mars", "metax", "qy", "other"])
    parser.add_argument("--baseline-pin", required=True)
    parser.add_argument("--candidate-pin", required=True)
    parser.add_argument("--iteration", required=True)
    parser.add_argument("--promotion-threshold", type=float, default=1.0)
    parser.add_argument("--exhausted", default="")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="infiniprofiler")
    subparsers = parser.add_subparsers(dest="command", required=True)

    normalize = subparsers.add_parser("normalize", help="normalize a raw capture")
    normalize.add_argument("--source", required=True, choices=["hctracer", "naive-sidecar"])
    normalize.add_argument("--input", type=Path, required=True)
    normalize.add_argument("--baseline", type=Path)
    normalize.add_argument("--mode", default=None)
    normalize.add_argument("--pid", type=int)
    normalize.add_argument("--output", type=Path, required=True)

    sample = subparsers.add_parser("sample-process", help="run the naive /proc sampler")
    sample.add_argument("--pid", type=int)
    sample.add_argument("--match", action="append", default=[])
    sample.add_argument("--duration-s", type=float, default=10.0)
    sample.add_argument("--interval-s", type=float, default=1.0)
    sample.add_argument("--output", type=Path, required=True)

    request = subparsers.add_parser("promotion-request", help="emit AutoHelix measure output")
    request.add_argument("--profile", type=Path, required=True)
    request.add_argument("--wall-compare", type=Path, required=True)
    request.add_argument("--output", type=Path, required=True)
    _add_identity_args(request)

    io_compat = subparsers.add_parser("io-compat", help="write IO promotion helper artifacts")
    io_compat.add_argument("--profile", type=Path, required=True)
    io_compat.add_argument("--wall-compare", type=Path, required=True)
    io_compat.add_argument("--out-dir", type=Path, required=True)
    io_compat.add_argument("--exhausted", default="")

    validate = subparsers.add_parser("validate", help="validate a normalized profile")
    validate.add_argument("--input", type=Path, required=True)
    return parser


def _parse_exhausted(raw: str) -> list:
    if not raw.strip():
        return []
    if raw.startswith("["):
        value = json.loads(raw)
        if not isinstance(value, list):
            raise ValueError("--exhausted JSON must be a list")
        return [str(item) for item in value]
    return [item.strip() for item in raw.split(",") if item.strip()]


def run_normalize(args: argparse.Namespace) -> int:
    if args.source == "hctracer":
        profile = normalize_hctracer(
            args.input,
            mode=args.mode or "attach",
            pid=args.pid,
            baseline_path=args.baseline,
        )
    else:
        profile = normalize_sidecar(args.input, baseline_path=args.baseline, mode=args.mode or "sidecar")
    profile.write_json(args.output)
    print(f"wrote {args.output}")
    print(f"host_signals={len(profile.host_signals)} scheduler_signals={len(profile.scheduler_signals)} warnings={len(profile.warnings)}")
    return 0


def run_sample(args: argparse.Namespace) -> int:
    pids = [args.pid] if args.pid else discover_pids(args.match)
    if len(pids) != 1:
        print(f"expected exactly one live pid, got {pids}", file=sys.stderr)
        return 2
    rows = sample_interval(
        pids[0],
        duration_s=args.duration_s,
        interval_s=args.interval_s,
    )
    write_jsonl(
        [dict(row, type="process_thread_sample") for row in rows],
        args.output,
    )
    print(f"wrote {args.output} rows={len(rows)} pid={pids[0]}")
    return 0


def _load_profile(path: Path) -> ProfileDocument:
    from dataclasses import fields

    from .models import Capture, EngineMetric, HostSignal, SchedulerSignal

    raw = _load_json(path)
    if raw.get("schema_version") != "infiniprofiler-profile-1":
        raise ValueError("unsupported profile schema_version")
    capture_raw = raw.get("capture")
    if not isinstance(capture_raw, dict):
        raise ValueError("profile.capture must be an object")
    capture = Capture(**capture_raw)

    profile = ProfileDocument(capture=capture)
    for key, model in (
        ("host_signals", HostSignal),
        ("scheduler_signals", SchedulerSignal),
        ("metrics", EngineMetric),
    ):
        for item in raw.get(key) or []:
            if isinstance(item, dict):
                allowed = {field.name for field in fields(model)}
                getattr(profile, key).append(model(**{k: v for k, v in item.items() if k in allowed}))
    profile.warnings = [str(item) for item in raw.get("warnings") or []]
    return profile


def run_request(args: argparse.Namespace) -> int:
    profile = _load_profile(args.profile)
    compare = _load_json(args.wall_compare)
    payload = promotion_request(
        profile,
        wall_compare=compare,
        campaign_id=args.campaign_id,
        model=args.model,
        host=args.host,
        device_track=args.device_track,
        baseline_pin=args.baseline_pin,
        candidate_pin=args.candidate_pin,
        iteration=args.iteration,
        promotion_threshold=args.promotion_threshold,
        exhausted=_parse_exhausted(args.exhausted),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output} top_path={payload['diagnosis']['top_path']}")
    return 0


def run_io_compat(args: argparse.Namespace) -> int:
    profile = _load_profile(args.profile)
    compare = _load_json(args.wall_compare)
    paths = write_io_compat(
        profile=profile,
        wall_compare=compare,
        out_dir=args.out_dir,
        exhausted=_parse_exhausted(args.exhausted),
    )
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


def run_validate(args: argparse.Namespace) -> int:
    profile = _load_profile(args.input)
    print(
        json.dumps(
            {
                "schema_version": profile.schema_version,
                "source": profile.capture.source,
                "host_signals": len(profile.host_signals),
                "scheduler_signals": len(profile.scheduler_signals),
                "metrics": len(profile.metrics),
                "warnings": profile.warnings,
            },
            indent=2,
        )
    )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "normalize": run_normalize,
        "sample-process": run_sample,
        "promotion-request": run_request,
        "io-compat": run_io_compat,
        "validate": run_validate,
    }
    return handlers[args.command](args)
