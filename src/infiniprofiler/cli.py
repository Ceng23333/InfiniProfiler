"""Command-line interface for normalization and promotion bridging."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from .contracts import (
    PROFILE_SCHEMA,
    REQUEST_SCHEMA,
    ContractValidationError,
    validate_profile_v2,
    validate_promotion_request_v2,
)
from .adapters.dlpti import normalize as normalize_dlpti
from .adapters.hctracer import normalize as normalize_hctracer
from .adapters.nsys_sqlite import normalize as normalize_nsys
from .capture import (
    CaptureError,
    CaptureSpec,
    capture_campaign,
    compare_capture_manifests,
    merge_capture,
)
from .models import NORMALIZE_SCHEMA, ProfileDocument
from .promotion.autohelix import promotion_request, write_io_compat
from .promotion.ranker import primary_wall_ratio, rank_paths
from .promotion.mechanism import (
    MechanismError,
    apply_mechanism_evaluation,
    build_promotion_request_v2,
)
from .rank_views import RankViewError, build_rank_views_from_capture
from .source_map import (
    SourceMapError,
    build_source_map,
    load_rules,
    write_source_map,
)
from .tools.sidecar import normalize as normalize_sidecar
from .tools.process_sampler import discover_pids, sample_interval, write_jsonl
from .timeline import attribute_trace


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
    normalize.add_argument(
        "--source",
        required=True,
        choices=["hctracer", "mctracer", "nsys", "dlpti", "naive-sidecar"],
    )
    normalize.add_argument("--input", type=Path, required=True)
    normalize.add_argument("--baseline", type=Path)
    normalize.add_argument(
        "--mode",
        default=None,
        help="capture mode; for --source dlpti defaults to wrap (not attach)",
    )
    normalize.add_argument("--pid", type=int)
    normalize.add_argument("--expected-ranks", type=int, default=1)
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
    request.add_argument("--timing-attribution", type=Path)
    request.add_argument("--source-map", type=Path)
    request.add_argument("--rank-views", type=Path)

    io_compat = subparsers.add_parser("io-compat", help="write IO promotion helper artifacts")
    io_compat.add_argument("--profile", type=Path, required=True)
    io_compat.add_argument("--wall-compare", type=Path, required=True)
    io_compat.add_argument("--out-dir", type=Path, required=True)
    io_compat.add_argument("--exhausted", default="")
    io_compat.add_argument("--timing-attribution", type=Path)
    io_compat.add_argument("--source-map", type=Path)
    io_compat.add_argument("--rank-views", type=Path)

    validate = subparsers.add_parser("validate", help="validate a normalized profile or request")
    validate.add_argument("--input", type=Path, required=True)
    validate.add_argument(
        "--kind",
        choices=["auto", "profile", "promotion-request"],
        default="auto",
    )

    capture = subparsers.add_parser("capture-campaign", help="run a transactional multi-rank capture")
    capture.add_argument("--spec", type=Path, required=True)
    capture.add_argument("--out-dir", type=Path, required=True)
    capture.add_argument("--hctracer-bin", default="hcTracer")
    capture.add_argument("--ready-seconds", type=float, default=2.0)
    capture.add_argument("--stop-wait-seconds", type=float, default=5.0)

    compare = subparsers.add_parser("compare-capture-manifests", help="compare baseline and candidate captures")
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)

    merge = subparsers.add_parser("merge-capture", help="normalize and merge one successful capture")
    merge.add_argument("--capture-dir", type=Path, required=True)
    merge.add_argument("--output", type=Path)

    attribute = subparsers.add_parser(
        "attribute-trace",
        help="stream one trace and emit exclusive timing attribution",
    )
    attribute.add_argument("--input", type=Path, required=True)
    attribute.add_argument("--output", type=Path, required=True)

    source_map_parser = subparsers.add_parser(
        "build-source-map",
        help="scan a source tree into source-map-1",
    )
    source_map_parser.add_argument("--source-root", type=Path, required=True)
    source_map_parser.add_argument("--source-pin", required=True)
    source_map_parser.add_argument("--rules", type=Path)
    source_map_parser.add_argument(
        "--required-symbol",
        action="append",
        required=True,
    )
    source_map_parser.add_argument("--output", type=Path, required=True)

    request_v2 = subparsers.add_parser(
        "promotion-request-v2",
        help="emit promotion-request-2 with a lever contract",
    )
    request_v2.add_argument("--profile", type=Path, required=True)
    request_v2.add_argument("--wall-compare", type=Path, required=True)
    request_v2.add_argument("--lever-contract", type=Path, required=True)
    request_v2.add_argument("--window-id", required=True)
    request_v2.add_argument("--profile-ref", required=True)
    request_v2.add_argument("--before-profile", type=Path)
    request_v2.add_argument("--after-profile", type=Path)
    request_v2.add_argument("--before-profile-ref", required=True)
    request_v2.add_argument("--after-profile-ref")
    request_v2.add_argument("--baseline-backend", default="baseline")
    request_v2.add_argument("--candidate-backend", default="candidate")
    request_v2.add_argument("--output", type=Path, required=True)
    _add_identity_args(request_v2)
    request_v2.add_argument("--timing-attribution", type=Path)
    request_v2.add_argument("--source-map", type=Path)
    request_v2.add_argument("--rank-views", type=Path)

    evaluate = subparsers.add_parser(
        "evaluate-mechanism",
        help="evaluate before/after mechanism movement and revalidate request",
    )
    evaluate.add_argument("--request", type=Path, required=True)
    evaluate.add_argument("--before-profile", type=Path, required=True)
    evaluate.add_argument("--after-profile", type=Path, required=True)
    evaluate.add_argument("--before-profile-ref", required=True)
    evaluate.add_argument("--after-profile-ref", required=True)
    evaluate.add_argument("--output", type=Path, required=True)

    rank_capture = subparsers.add_parser(
        "rank-capture",
        help="aggregate per-rank TP imbalance and straggler facts",
    )
    rank_capture.add_argument("--capture-dir", type=Path, required=True)
    rank_capture.add_argument("--output", type=Path)
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


def _print_timing_summary(profile: ProfileDocument) -> None:
    timing = profile.timing_attribution
    if timing is None:
        print(
            f"host_signals={len(profile.host_signals)} metrics={len(profile.metrics)} "
            f"warnings={len(profile.warnings)} timing_attribution=missing"
        )
        return
    print(
        f"host_signals={len(profile.host_signals)} metrics={len(profile.metrics)} "
        f"warnings={len(profile.warnings)} "
        f"api_submit_us={timing.api_submit_us:.3f} "
        f"gpu_execution_us={timing.gpu_execution_us:.3f} "
        f"sync_wait_us={timing.sync_wait_us:.3f} "
        f"graph_launch_us={timing.graph_launch_us:.3f}"
    )


def run_normalize(args: argparse.Namespace) -> int:
    if args.source in ("hctracer", "mctracer"):
        profile = normalize_hctracer(
            args.input,
            source_name=args.source,
            mode=args.mode or "attach",
            pid=args.pid,
            baseline_path=args.baseline,
        )
    elif args.source == "nsys":
        profile = normalize_nsys(
            args.input,
            expected_ranks=args.expected_ranks,
            mode=args.mode or "nsys",
            baseline_path=args.baseline,
        )
    elif args.source == "dlpti":
        profile = normalize_dlpti(
            args.input,
            source_name=args.source,
            mode=args.mode or "wrap",
            pid=args.pid,
            baseline_path=args.baseline,
        )
    else:
        profile = normalize_sidecar(
            args.input, baseline_path=args.baseline, mode=args.mode or "sidecar"
        )
    profile.write_json(args.output)
    print(f"wrote {args.output}")
    _print_timing_summary(profile)
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

    from .models import Capture, EngineMetric, HostSignal, TimingAttributionFields

    raw = _load_json(path)
    schema = raw.get("schema_version")
    if schema not in (NORMALIZE_SCHEMA, "infiniprofiler-normalize-1"):
        raise ValueError(
            f"unsupported profile schema_version for normalize load: {schema}"
        )
    capture_raw = raw.get("capture")
    if not isinstance(capture_raw, dict):
        raise ValueError("profile.capture must be an object")
    capture = Capture(**capture_raw)

    profile = ProfileDocument(capture=capture)
    for key, model in (
        ("host_signals", HostSignal),
        ("metrics", EngineMetric),
    ):
        for item in raw.get(key) or []:
            if isinstance(item, dict):
                allowed = {field.name for field in fields(model)}
                getattr(profile, key).append(
                    model(**{k: v for k, v in item.items() if k in allowed})
                )
    timing_raw = raw.get("timing_attribution")
    if isinstance(timing_raw, dict):
        profile.timing_attribution = TimingAttributionFields.from_mapping(timing_raw)
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
        graph_timing=_load_json(args.timing_attribution) if args.timing_attribution else None,
        graph_source=_load_json(args.source_map) if args.source_map else None,
        graph_ranks=_load_json(args.rank_views) if args.rank_views else None,
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
        graph_timing=_load_json(args.timing_attribution) if args.timing_attribution else None,
        graph_source=_load_json(args.source_map) if args.source_map else None,
        graph_ranks=_load_json(args.rank_views) if args.rank_views else None,
    )
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


def run_validate(args: argparse.Namespace) -> int:
    raw = _load_json(args.input)
    kind = args.kind
    schema = str(raw.get("schema_version", ""))
    if kind == "auto":
        if schema == PROFILE_SCHEMA:
            kind = "profile"
        elif schema == REQUEST_SCHEMA:
            kind = "promotion-request"
        else:
            kind = "profile"

    if kind == "promotion-request":
        if schema not in ("promotion-request-1", REQUEST_SCHEMA):
            raise ValueError(f"unsupported promotion request schema_version: {schema}")
        if schema == REQUEST_SCHEMA:
            validate_promotion_request_v2(raw)
        summary = {
            "schema_version": schema,
            "kind": "promotion-request",
            "top_path": raw.get("diagnosis", {}).get("top_path"),
            "stop_before_promote": raw.get("stop_before_promote"),
            "mechanism_status": raw.get("mechanism_check", {}).get("status", "NOT_RUN"),
        }
    else:
        if schema == PROFILE_SCHEMA:
            profile = validate_profile_v2(raw)
            summary = {
                "schema_version": PROFILE_SCHEMA,
                "kind": "profile",
                "source": profile.capture.source,
                "window_id": profile.window.window_id,
                "alignment": profile.window.alignment,
                "campaign_id": profile.workload.campaign_id,
                "ranks": len(profile.ranks),
                "unpaired_events": profile.timing.unpaired_event_count,
                "api_submit_us": profile.timing.api_submit_us,
                "gpu_execution_us": profile.timing.gpu_execution_us,
                "sync_wait_us": profile.timing.sync_wait_us,
                "warnings": profile.warnings,
            }
        elif schema == NORMALIZE_SCHEMA:
            profile = _load_profile(args.input)
            timing = profile.timing_attribution
            summary = {
                "schema_version": profile.schema_version,
                "kind": "profile",
                "source": profile.capture.source,
                "host_signals": len(profile.host_signals),
                "metrics": len(profile.metrics),
                "timing_attribution": timing.to_dict() if timing else None,
                "warnings": profile.warnings,
            }
        else:
            raise ValueError(f"unsupported profile schema_version: {schema}")
    print(json.dumps(summary, indent=2))
    return 0


def run_capture(args: argparse.Namespace) -> int:
    spec = CaptureSpec.from_file(args.spec)
    capture = capture_campaign(
        spec,
        args.out_dir,
        hctracer_bin=args.hctracer_bin,
        ready_seconds=args.ready_seconds,
        stop_wait_seconds=args.stop_wait_seconds,
    )
    print(json.dumps(capture.to_dict(), indent=2, sort_keys=True))
    return 0 if capture.status == "SUCCEEDED" else 1


def run_compare_capture(args: argparse.Namespace) -> int:
    result = compare_capture_manifests(args.baseline, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output} status={result['status']}")
    return 0 if result["status"] == "MATCHED" else 1


def run_merge_capture(args: argparse.Namespace) -> int:
    payload = merge_capture(args.capture_dir, args.output)
    destination = args.output or args.capture_dir / "capture_merge.json"
    print(f"wrote {destination} status={payload['status']} ranks={payload['rank_count']}")
    return 0


def run_attribute_trace(args: argparse.Namespace) -> int:
    timeline = attribute_trace(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(timeline.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output} evidence_mode={timeline.evidence_mode}")
    return 0


def run_rank_capture(args: argparse.Namespace) -> int:
    output = args.output or args.capture_dir / "rank_views.json"
    result = build_rank_views_from_capture(args.capture_dir, output)
    print(
        f"wrote {output} status={result.status} ranks={len(result.rank_views)} "
        f"boundary_status={result.boundary_status}"
    )
    return 0


def run_build_source_map(args: argparse.Namespace) -> int:
    rules = load_rules(args.rules) if args.rules else None
    document = build_source_map(
        args.source_root,
        source_pin=args.source_pin,
        rules=rules,
        required_symbols=args.required_symbol,
    )
    write_source_map(document, args.output)
    print(
        f"wrote {args.output} files={document['scanned_file_count']} "
        f"symbols={len(document['symbols'])} providers={len(document['providers'])}"
    )
    return 0


def run_request_v2(args: argparse.Namespace) -> int:
    profile = _load_profile(args.profile)
    after = _load_json(args.after_profile) if args.after_profile else None
    payload = build_promotion_request_v2(
        profile,
        wall_compare=_load_json(args.wall_compare),
        lever_contract=_load_json(args.lever_contract),
        window_id=args.window_id,
        campaign_id=args.campaign_id,
        model=args.model,
        host=args.host,
        device_track=args.device_track,
        baseline_pin=args.baseline_pin,
        candidate_pin=args.candidate_pin,
        iteration=args.iteration,
        profile_ref=args.profile_ref,
        before_profile_ref=args.before_profile_ref,
        after_profile_ref=args.after_profile_ref or "",
        before=_load_json(args.before_profile) if args.before_profile else None,
        after=after,
        baseline_backend=args.baseline_backend,
        candidate_backend=args.candidate_backend,
        exhausted=_parse_exhausted(args.exhausted),
        graph_timing=_load_json(args.timing_attribution) if args.timing_attribution else None,
        graph_source=_load_json(args.source_map) if args.source_map else None,
        graph_ranks=_load_json(args.rank_views) if args.rank_views else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"wrote {args.output} top_path={payload['diagnosis']['top_path']} "
        f"mechanism={payload['mechanism_check']['status']}"
    )
    return 0


def run_evaluate_mechanism(args: argparse.Namespace) -> int:
    payload = apply_mechanism_evaluation(
        _load_json(args.request),
        before=_load_json(args.before_profile),
        after=_load_json(args.after_profile),
        before_profile_ref=args.before_profile_ref,
        after_profile_ref=args.after_profile_ref,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output} mechanism={payload['mechanism_check']['status']}")
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
        "capture-campaign": run_capture,
        "compare-capture-manifests": run_compare_capture,
        "merge-capture": run_merge_capture,
        "attribute-trace": run_attribute_trace,
        "rank-capture": run_rank_capture,
        "build-source-map": run_build_source_map,
        "promotion-request-v2": run_request_v2,
        "evaluate-mechanism": run_evaluate_mechanism,
    }
    try:
        return handlers[args.command](args)
    except ContractValidationError as exc:
        for error in exc.errors:
            print(f"contract error: {error}", file=sys.stderr)
        return 2
    except CaptureError as exc:
        print(f"capture error: {exc}", file=sys.stderr)
        return 1
    except RankViewError as exc:
        print(f"rank-view error: {exc}", file=sys.stderr)
        return 1
    except SourceMapError as exc:
        print(f"source-map error: {exc}", file=sys.stderr)
        return 2
    except MechanismError as exc:
        print(f"mechanism error: {exc}", file=sys.stderr)
        return 2
