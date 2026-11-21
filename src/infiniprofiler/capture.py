"""Transactional multi-rank hcTracer campaign capture."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from .adapters.hctracer import normalize as normalize_hctracer


SPEC_SCHEMA = "infiniprofiler-capture-spec-1"
MANIFEST_SCHEMA = "infiniprofiler-capture-manifest-1"
MERGE_SCHEMA = "infiniprofiler-capture-merge-1"
DEFAULT_START_SKEW_MS = 50.0
DEFAULT_DURATION_RATIO_RANGE = (0.95, 1.05)
CONTROL_T = b"\x14"


class CaptureError(RuntimeError):
    """A campaign capture could not produce promotion-grade evidence."""


def _required(mapping: Mapping[str, Any], keys: Iterable[str], path: str, errors: List[str]) -> None:
    for key in keys:
        value = mapping.get(key)
        if value is None or value == "":
            errors.append(f"{path}.{key} is required")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class CaptureSpec:
    role: str
    window_id: str
    campaign_id: str
    model: str
    baseline_pin: str
    candidate_pin: str
    tensor_parallel: int
    wall_workload_id: str
    wall_compare_ref: str
    warmup_excluded: bool
    client_command: str
    concurrency: int
    prompt_lengths: List[int]
    output_lengths: List[int]
    clock_policy: str
    profiles_source_hash: str
    rank_pids: Dict[int, int]
    batch_size: Optional[int] = None

    @classmethod
    def from_file(cls, path: Path) -> "CaptureSpec":
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CaptureError(f"cannot read capture spec {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise CaptureError("capture spec must be a JSON object")
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CaptureSpec":
        errors: List[str] = []
        if raw.get("schema_version") != SPEC_SCHEMA:
            errors.append(f"schema_version must be {SPEC_SCHEMA}")
        _required(
            raw,
            (
                "role",
                "window_id",
                "campaign_id",
                "model",
                "baseline_pin",
                "candidate_pin",
                "tensor_parallel",
                "wall_workload_id",
                "wall_compare_ref",
                "client_command",
                "concurrency",
                "prompt_lengths",
                "output_lengths",
                "clock_policy",
                "profiles_source_hash",
                "rank_pids",
            ),
            "spec",
            errors,
        )
        if raw.get("role") not in ("baseline", "candidate"):
            errors.append("spec.role must be baseline or candidate")
        if raw.get("warmup_excluded") is not True:
            errors.append("spec.warmup_excluded must be true")

        tp = raw.get("tensor_parallel")
        concurrency = raw.get("concurrency")
        batch_size = raw.get("batch_size")
        for name, value, minimum in (
            ("tensor_parallel", tp, 1),
            ("concurrency", concurrency, 0),
            ("batch_size", batch_size, 1),
        ):
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                errors.append(f"spec.{name} must be an integer >= {minimum}")

        for name in ("prompt_lengths", "output_lengths"):
            values = raw.get(name)
            if not isinstance(values, list) or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in values
            ):
                errors.append(f"spec.{name} must be a list of non-negative integers")

        rank_pids_raw = raw.get("rank_pids")
        rank_pids: Dict[int, int] = {}
        if not isinstance(rank_pids_raw, dict) or not rank_pids_raw:
            errors.append("spec.rank_pids must be a non-empty object")
        else:
            seen_pids = set()
            for key, value in rank_pids_raw.items():
                try:
                    rank = int(key)
                    pid = int(value)
                except (TypeError, ValueError):
                    errors.append(f"spec.rank_pids.{key} must map a rank to an integer pid")
                    continue
                if rank < 0 or pid <= 0:
                    errors.append(f"spec.rank_pids.{key} has invalid rank or pid")
                    continue
                if rank in rank_pids:
                    errors.append(f"spec.rank_pids has duplicate rank {rank}")
                if pid in seen_pids:
                    # Concurrent hcTracer attach to the same pid is unsafe and
                    # cannot be attributed to independent ranks.
                    errors.append(f"spec.rank_pids maps multiple ranks to pid {pid}")
                seen_pids.add(pid)
                rank_pids[rank] = pid

        expected_ranks = tp if isinstance(tp, int) and not isinstance(tp, bool) else -1
        if isinstance(rank_pids_raw, dict) and len(rank_pids) != expected_ranks:
            errors.append(f"spec.rank_pids must contain exactly {expected_ranks} ranks")

        if errors:
            raise CaptureError("; ".join(errors))
        return cls(
            role=str(raw["role"]),
            window_id=str(raw["window_id"]),
            campaign_id=str(raw["campaign_id"]),
            model=str(raw["model"]),
            baseline_pin=str(raw["baseline_pin"]),
            candidate_pin=str(raw["candidate_pin"]),
            tensor_parallel=int(tp),
            wall_workload_id=str(raw["wall_workload_id"]),
            wall_compare_ref=str(raw["wall_compare_ref"]),
            warmup_excluded=True,
            client_command=str(raw["client_command"]),
            concurrency=int(concurrency),
            prompt_lengths=list(raw["prompt_lengths"]),
            output_lengths=list(raw["output_lengths"]),
            clock_policy=str(raw["clock_policy"]),
            profiles_source_hash=str(raw["profiles_source_hash"]),
            rank_pids=rank_pids,
            batch_size=int(batch_size) if batch_size is not None else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["rank_pids"] = {str(rank): pid for rank, pid in sorted(self.rank_pids.items())}
        payload["client_command_sha256"] = _sha256(self.client_command)
        return payload


@dataclass
class RankStatus:
    rank: int
    pid: int
    status: str
    returncode: Optional[int] = None
    raw_files: List[str] = field(default_factory=list)
    raw_bytes: int = 0
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CampaignCapture:
    spec: CaptureSpec
    status: str
    started_epoch_ns: int
    ended_epoch_ns: int
    ranks: List[RankStatus]
    client_returncode: Optional[int] = None
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": MANIFEST_SCHEMA,
            "status": self.status,
            "started_epoch_ns": self.started_epoch_ns,
            "ended_epoch_ns": self.ended_epoch_ns,
            "duration_ns": max(0, self.ended_epoch_ns - self.started_epoch_ns),
            "client_returncode": self.client_returncode,
            "detail": self.detail,
            "spec": self.spec.to_dict(),
            "ranks": [rank.to_dict() for rank in self.ranks],
        }


TracerCommandBuilder = Callable[[int, int], Sequence[str]]


def default_tracer_command(hctracer_bin: str) -> TracerCommandBuilder:
    def build(rank: int, pid: int) -> Sequence[str]:
        del rank
        return [
            hctracer_bin,
            "--hctx",
            "--attach",
            str(pid),
            "--odname",
            "raw",
            "--name",
            "tracer_out",
        ]

    return build


def _raw_files(rank_dir: Path) -> List[Path]:
    raw_dir = rank_dir / "raw"
    if not raw_dir.exists():
        return []
    files = sorted(raw_dir.rglob("tracer_out*.json"))
    files += [path for path in sorted(raw_dir.rglob("*.json")) if path not in files and path.name != "attach_tracer.log"]
    return files


def _stop_process(process: subprocess.Popen, stop_wait_s: float) -> None:
    if process.poll() is not None:
        if process.stdin:
            process.stdin.close()
        return
    try:
        if process.stdin:
            process.stdin.write(CONTROL_T)
            process.stdin.flush()
            process.stdin.close()
    except (BrokenPipeError, OSError):
        pass
    try:
        process.wait(timeout=max(0.1, stop_wait_s))
        return
    except subprocess.TimeoutExpired:
        pass
    process.terminate()
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _collect_rank(
    rank_dir: Path,
    *,
    rank: int,
    pid: int,
    returncode: Optional[int],
    client_failed: bool,
) -> RankStatus:
    files = _raw_files(rank_dir)
    raw_bytes = sum(path.stat().st_size for path in files if path.exists())
    minimum_bytes = int(os.environ.get("MIN_ATTACH_BYTES", "65536"))
    status = "SUCCEEDED"
    detail = ""
    if returncode != 0:
        status = "FAILED_CAPTURE"
        detail = f"hcTracer returncode={returncode}"
    elif not files:
        status = "FAILED_CAPTURE"
        detail = "no tracer_out*.json produced"
    elif raw_bytes < minimum_bytes:
        status = "FAILED_CAPTURE"
        detail = f"capture too small: {raw_bytes} < {minimum_bytes} bytes"
    if client_failed and status == "SUCCEEDED":
        status = "CLIENT_WINDOW_FAILED"
        detail = "tracer stopped but client window failed"
    return RankStatus(
        rank=rank,
        pid=pid,
        status=status,
        returncode=returncode,
        raw_files=[str(path) for path in files],
        raw_bytes=raw_bytes,
        detail=detail,
    )


def capture_campaign(
    spec: CaptureSpec,
    out_dir: Path,
    *,
    hctracer_bin: str = "hcTracer",
    ready_seconds: float = 2.0,
    stop_wait_seconds: float = 5.0,
    tracer_command: Optional[TracerCommandBuilder] = None,
) -> CampaignCapture:
    """Attach all ranks, run exactly one client window, and stop every tracer."""
    if tracer_command is None:
        tracer_command = default_tracer_command(hctracer_bin)
    if ready_seconds <= 0 or stop_wait_seconds <= 0:
        raise CaptureError("ready_seconds and stop_wait_seconds must be positive")

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    processes: Dict[int, subprocess.Popen] = {}
    finalized_capture: Optional[CampaignCapture] = None
    rank_statuses: Dict[int, RankStatus] = {
        rank: RankStatus(rank=rank, pid=pid, status="ATTACHING")
        for rank, pid in sorted(spec.rank_pids.items())
    }
    started_ns = time.time_ns()
    running_manifest = CampaignCapture(
        spec=spec,
        status="RUNNING",
        started_epoch_ns=started_ns,
        ended_epoch_ns=started_ns,
        ranks=list(rank_statuses.values()),
    )
    manifest_path = out_dir / "window_manifest.json"
    _write_json_atomic(manifest_path, running_manifest.to_dict())

    def finalize(
        *,
        status: str,
        detail: str,
        client_returncode: Optional[int] = None,
    ) -> CampaignCapture:
        ended_ns = time.time_ns()
        capture = CampaignCapture(
            spec=spec,
            status=status,
            started_epoch_ns=started_ns,
            ended_epoch_ns=ended_ns,
            ranks=[rank_statuses[rank] for rank in sorted(rank_statuses)],
            client_returncode=client_returncode,
            detail=detail,
        )
        _write_json_atomic(manifest_path, capture.to_dict())
        _write_json_atomic(out_dir / "capture_spec.json", spec.to_dict())
        return capture

    try:
        for rank, pid in sorted(spec.rank_pids.items()):
            rank_dir = out_dir / f"rank={rank:02d}"
            (rank_dir / "raw").mkdir(parents=True, exist_ok=True)
            log_path = rank_dir / "raw" / "attach_tracer.log"
            log_handle = log_path.open("w", encoding="utf-8")
            try:
                process = subprocess.Popen(
                    list(tracer_command(rank, pid)),
                    cwd=str(rank_dir),
                    stdin=subprocess.PIPE,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                )
                processes[rank] = process
                rank_statuses[rank].status = "ATTACHED"
            except OSError as exc:
                rank_statuses[rank].status = "FAILED_ATTACH"
                rank_statuses[rank].detail = str(exc)
                raise CaptureError(f"rank {rank} attach failed: {exc}") from exc
            finally:
                log_handle.close()

        time.sleep(ready_seconds)
        failed = [
            rank
            for rank, process in processes.items()
            if process.poll() is not None
        ]
        if failed:
            for rank in failed:
                returncode = processes[rank].wait()
                rank_statuses[rank] = _collect_rank(
                    out_dir / f"rank={rank:02d}",
                    rank=rank,
                    pid=spec.rank_pids[rank],
                    returncode=returncode,
                    client_failed=False,
                )
                rank_statuses[rank].status = "FAILED_ATTACH"
                rank_statuses[rank].detail = rank_statuses[rank].detail or "hcTracer exited before client window"
            raise CaptureError(f"rank attach failed before client window: ranks={failed}")

        client = subprocess.run(["bash", "-lc", spec.client_command])
        client_failed = client.returncode != 0
        for rank, process in sorted(processes.items()):
            _stop_process(process, stop_wait_seconds)
            rank_statuses[rank] = _collect_rank(
                out_dir / f"rank={rank:02d}",
                rank=rank,
                pid=spec.rank_pids[rank],
                returncode=process.returncode,
                client_failed=client_failed,
            )

        if client_failed:
            capture = finalize(
                status="CLIENT_WINDOW_FAILED",
                detail=f"client returncode={client.returncode}",
                client_returncode=client.returncode,
            )
            finalized_capture = capture
            raise CaptureError("client window failed; no ranked request emitted") from None
        failed_ranks = [item.status for item in rank_statuses.values() if item.status != "SUCCEEDED"]
        if failed_ranks:
            capture = finalize(
                status="FAILED_RANK_CAPTURE",
                detail=f"rank failures={failed_ranks}",
                client_returncode=client.returncode,
            )
            finalized_capture = capture
            raise CaptureError("one or more rank captures failed; no ranked request emitted") from None
        return finalize(
            status="SUCCEEDED",
            detail="all required ranks captured one shared client window",
            client_returncode=client.returncode,
        )
    except (CaptureError, KeyboardInterrupt):
        for process in processes.values():
            if process.poll() is None:
                _stop_process(process, stop_wait_seconds)
        for rank, process in processes.items():
            current = rank_statuses[rank]
            if current.status in ("ATTACHED", "ATTACHING"):
                rank_statuses[rank] = _collect_rank(
                    out_dir / f"rank={rank:02d}",
                    rank=rank,
                    pid=spec.rank_pids[rank],
                    returncode=process.returncode,
                    client_failed=True,
                )
                if current.status == "ATTACHED":
                    rank_statuses[rank].status = "STOPPED_DURING_FAILURE"
                elif rank_statuses[rank].status == "SUCCEEDED":
                    rank_statuses[rank].status = "STOPPED_DURING_FAILURE"
        if finalized_capture is not None:
            raise
        finalize(
            status="FAILED_RANK_ATTACH",
            detail="capture aborted transactionally; no ranked request emitted",
        )
        raise


def read_manifest(path: Path) -> Dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CaptureError(f"cannot read manifest {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise CaptureError(f"manifest {path} is not an object")
    if document.get("schema_version") != MANIFEST_SCHEMA:
        raise CaptureError(f"manifest {path} has unsupported schema_version")
    return document


def _identity(spec: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "campaign_id": spec.get("campaign_id"),
        "model": spec.get("model"),
        "baseline_pin": spec.get("baseline_pin"),
        "candidate_pin": spec.get("candidate_pin"),
        "tensor_parallel": spec.get("tensor_parallel"),
        "wall_workload_id": spec.get("wall_workload_id"),
        "wall_compare_ref": spec.get("wall_compare_ref"),
        "warmup_excluded": spec.get("warmup_excluded"),
        "concurrency": spec.get("concurrency"),
        "prompt_lengths": spec.get("prompt_lengths"),
        "output_lengths": spec.get("output_lengths"),
        "batch_size": spec.get("batch_size"),
        "clock_policy": spec.get("clock_policy"),
        "profiles_source_hash": spec.get("profiles_source_hash"),
        "client_command_sha256": spec.get("client_command_sha256"),
    }


def compare_capture_manifests(
    baseline_path: Path,
    candidate_path: Path,
    *,
    duration_tolerance: Sequence[float] = DEFAULT_DURATION_RATIO_RANGE,
) -> Dict[str, Any]:
    baseline = read_manifest(baseline_path)
    candidate = read_manifest(candidate_path)
    errors: List[str] = []
    if baseline.get("status") != "SUCCEEDED":
        errors.append(f"baseline capture status is {baseline.get('status')}, not SUCCEEDED")
    if candidate.get("status") != "SUCCEEDED":
        errors.append(f"candidate capture status is {candidate.get('status')}, not SUCCEEDED")

    baseline_identity = _identity(baseline.get("spec", {}))
    candidate_identity = _identity(candidate.get("spec", {}))
    for key, baseline_value in baseline_identity.items():
        candidate_value = candidate_identity.get(key)
        if candidate_value != baseline_value:
            errors.append(f"identity mismatch: {key} baseline={baseline_value!r} candidate={candidate_value!r}")

    baseline_ranks = baseline.get("ranks", [])
    candidate_ranks = candidate.get("ranks", [])
    baseline_ids = sorted(rank.get("rank") for rank in baseline_ranks)
    candidate_ids = sorted(rank.get("rank") for rank in candidate_ranks)
    if baseline_ids != candidate_ids:
        errors.append(f"rank ids differ: baseline={baseline_ids} candidate={candidate_ids}")

    baseline_duration = float(baseline.get("duration_ns", 0))
    candidate_duration = float(candidate.get("duration_ns", 0))
    if baseline_duration <= 0 or candidate_duration <= 0:
        errors.append("both captures must have positive client-window durations")
    else:
        ratio = candidate_duration / baseline_duration
        if not duration_tolerance[0] <= ratio <= duration_tolerance[1]:
            errors.append(
                f"window duration ratio {ratio:.6f} outside "
                f"{duration_tolerance[0]}-{duration_tolerance[1]}"
            )

    result = {
        "schema_version": "infiniprofiler-capture-comparison-1",
        "baseline_manifest": str(baseline_path),
        "candidate_manifest": str(candidate_path),
        "status": "MATCHED" if not errors else "MISMATCHED",
        "errors": errors,
        "duration_ratio": candidate_duration / baseline_duration if baseline_duration else None,
        "identity": baseline_identity,
    }
    return result


def merge_capture(out_dir: Path, output_path: Optional[Path] = None) -> Dict[str, Any]:
    """Normalize each rank independently and write a compact manifest merge."""
    out_dir = out_dir.resolve()
    manifest = read_manifest(out_dir / "window_manifest.json")
    if manifest.get("status") != "SUCCEEDED":
        raise CaptureError(
            f"refusing to merge status={manifest.get('status')}; only SUCCEEDED captures are mergeable"
        )

    rank_summaries: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for rank_status in manifest.get("ranks", []):
        rank = int(rank_status["rank"])
        pid = int(rank_status["pid"])
        trace_paths = [Path(path) for path in rank_status.get("raw_files", [])]
        trace_paths = [path for path in trace_paths if path.is_file()]
        if not trace_paths:
            raise CaptureError(f"rank {rank} has no readable raw trace")
        profile = normalize_hctracer(trace_paths[0], source_name="hctracer", mode="attach", pid=pid)
        rank_summaries.append(
            {
                "rank": rank,
                "pid": pid,
                "raw_trace": str(trace_paths[0]),
                "raw_bytes": int(rank_status.get("raw_bytes", 0)),
                "event_count": profile.capture.provenance.get("event_count", 0),
                "host_totals_us": {
                    signal.family: signal.candidate for signal in profile.host_signals
                },
                "scheduler_signals": [signal.to_dict() for signal in profile.scheduler_signals],
                "metrics": [metric.to_dict() for metric in profile.metrics],
            }
        )
        warnings.extend(f"rank={rank}:{warning}" for warning in profile.warnings)

    payload = {
        "schema_version": MERGE_SCHEMA,
        "status": "COMPLETE",
        "manifest": str(out_dir / "window_manifest.json"),
        "window_id": manifest.get("spec", {}).get("window_id"),
        "role": manifest.get("spec", {}).get("role"),
        "rank_count": len(rank_summaries),
        "ranks": rank_summaries,
        "warnings": warnings,
    }
    destination = output_path or out_dir / "capture_merge.json"
    _write_json_atomic(Path(destination), payload)
    return payload
