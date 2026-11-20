"""Emit the deterministic AutoHelix measure request and IO-compatible facts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..models import ProfileDocument
from .ranker import LEVERS, primary_wall_ratio, rank_paths


REQUEST_SCHEMA = "promotion-request-1"


def _dump(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def promotion_request(
    profile: ProfileDocument,
    *,
    wall_compare: Dict[str, Any],
    campaign_id: str,
    model: str,
    host: str,
    device_track: str,
    baseline_pin: str,
    candidate_pin: str,
    iteration: str,
    promotion_threshold: float = 1.0,
    exhausted: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    wall_ratio = primary_wall_ratio(wall_compare)
    backlog = rank_paths(profile, wall_ratio=wall_ratio, exhausted=exhausted)
    top = backlog["ranked"][0]

    return {
        "schema_version": REQUEST_SCHEMA,
        "controller": "autohelix",
        "stop_before_promote": True,
        "campaign": {
            "campaign_id": campaign_id,
            "model": model,
            "host": host,
            "device_track": device_track,
            "baseline_backend": "vllm",
            "candidate_backend": "infinilm",
            "baseline_pin": baseline_pin,
            "candidate_pin": candidate_pin,
            "iteration": iteration,
        },
        "observation": {
            "profile_schema_version": profile.schema_version,
            "profile_source": profile.capture.source,
            "profile_mode": profile.capture.mode,
            "profile_warnings": profile.warnings,
            "wall_ratio": wall_ratio,
            "wall_compare": wall_compare,
        },
        "gate": {
            "metric": "unprofiled_primary_wall",
            "surpass_threshold": promotion_threshold,
            "profiled_result_can_promote": False,
        },
        "diagnosis": backlog,
        "promote": {
            "one_path": top["path_id"],
            "lever_hint": top["lever"],
            "may_apply": False,
            "must_reobserve_unprofiled": True,
        },
    }


def write_io_compat(
    *,
    profile: ProfileDocument,
    wall_compare: Dict[str, Any],
    out_dir: Path,
    exhausted: Optional[Sequence[str]] = None,
) -> Dict[str, Path]:
    """Write namespaced compare/backlog data compatible with IO promo parsing."""
    wall_ratio = primary_wall_ratio(wall_compare)
    backlog = rank_paths(profile, wall_ratio=wall_ratio, exhausted=exhausted)
    host_signals: List[Dict[str, Any]] = [item.to_dict() for item in profile.host_signals]
    compare = {
        "ok": True,
        "status": "NORMALIZED",
        "vendor_profiler": profile.capture.source,
        "parser": "infiniprofiler",
        "host_signals": host_signals,
        # Unknown optional fields are ignored safely by promo-facts-1 consumers.
        "scheduler_signals": [item.to_dict() for item in profile.scheduler_signals],
        "metrics": [item.to_dict() for item in profile.metrics],
        "warnings": profile.warnings,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    compare_path = out_dir / "compare.json"
    backlog_path = out_dir / "promotion_backlog.json"
    request_path = out_dir / "autohelix_request.json"
    _dump(compare_path, compare)
    _dump(backlog_path, backlog)
    _dump(request_path, compare | backlog)
    return {
        "compare": compare_path,
        "backlog": backlog_path,
        "request": request_path,
    }
