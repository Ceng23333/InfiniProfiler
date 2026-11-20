"""A deliberately naive /proc sampler for host CPU scheduler evidence."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def _read_cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return " ".join(part.decode("utf-8", "replace") for part in raw.split(b"\0") if part)


def discover_pids(patterns: Iterable[str]) -> List[int]:
    needles = [item.lower() for item in patterns]
    matches: List[int] = []
    proc = Path("/proc")
    if not proc.exists():
        return matches
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        command = _read_cmdline(pid).lower()
        if command and any(needle in command for needle in needles):
            matches.append(pid)
    return sorted(matches)


def _parse_status(path: Path) -> Tuple[int, int]:
    voluntary = 0
    involuntary = 0
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            key, separator, value = line.partition(":")
            if not separator:
                continue
            value = value.strip()
            if key == "voluntary_ctxt_switches":
                voluntary = int(value)
            elif key == "nonvoluntary_ctxt_switches":
                involuntary = int(value)
    except (OSError, ValueError):
        pass
    return voluntary, involuntary


def _parse_stat_task(path: Path) -> Optional[Tuple[str, Dict[str, int]]]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    comm_start = text.find("(")
    comm_end = text.rfind(")")
    if comm_start < 0 or comm_end <= comm_start:
        return None
    fields = text[comm_end + 2 :].split()
    if len(fields) < 36:
        return None

    def field(number: int) -> int:
        # /proc/[pid]/stat fields are one-based; the list starts at state
        # because pid and comm (with parentheses) were removed above.
        return int(fields[number - 3])

    return text[:comm_start].strip(), {
        "utime": field(14),
        "stime": field(15),
        "num_threads": field(20),
    }


def sample_process(pid: int) -> Dict[str, List[Dict[str, float]]]:
    """Read cumulative CPU ticks and context switches for all tasks."""
    output: Dict[str, List[Dict[str, float]]] = {}
    tasks = Path(f"/proc/{pid}/task")
    for task_entry in tasks.iterdir() if tasks.exists() else []:
        parsed = _parse_stat_task(task_entry / "stat")
        if parsed is None:
            continue
        tid, values = parsed
        voluntary, involuntary = _parse_status(task_entry / "status")
        values["vcs"] = voluntary
        values["ivcs"] = involuntary
        output.setdefault("threads", []).append({"tid": int(tid), **values})
    output["threads"].sort(key=lambda item: int(item["tid"]))
    return output


def sample_interval(
    pid: int,
    *,
    duration_s: float,
    interval_s: float,
) -> List[Dict[str, float]]:
    """Sample CPU deltas for one process and emit one row per interval."""
    if duration_s <= 0 or interval_s <= 0:
        raise ValueError("duration_s and interval_s must be positive")
    clock_ticks = os.sysconf("SC_CLK_TCK")
    rows: List[Dict[str, float]] = []
    deadline = time.monotonic() + duration_s
    previous = sample_process(pid)
    previous_at = time.monotonic()

    while time.monotonic() < deadline:
        remaining = min(interval_s, max(0.0, deadline - time.monotonic()))
        time.sleep(remaining)
        current_at = time.monotonic()
        current = sample_process(pid)
        old = {int(item["tid"]): item for item in previous["threads"]}
        for thread in current["threads"]:
            tid = int(thread["tid"])
            before = old.get(tid)
            if not before:
                continue
            elapsed = max(current_at - previous_at, 1e-9)
            cpu_ticks = (thread["utime"] - before["utime"]) + (
                thread["stime"] - before["stime"]
            )
            rows.append(
                {
                    "pid": pid,
                    "tid": tid,
                    "cpu_percent": 100.0 * cpu_ticks / (clock_ticks * elapsed),
                    "vcs": thread["vcs"] - before["vcs"],
                    "ivcs": thread["ivcs"] - before["ivcs"],
                }
            )
        previous = current
        previous_at = current_at
    return rows


def write_jsonl(rows: Iterable[Dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
