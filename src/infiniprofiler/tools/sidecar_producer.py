"""Minimal producer API for engines without HCTX instrumentation."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Union

from ..scheduler import CANONICAL_SPANS


class SidecarSpanRecorder:
    """Append explicit engine ranges to a JSONL file."""

    def __init__(
        self,
        path: Union[str, Path],
        *,
        rank: Optional[int] = None,
        window_id: str,
    ) -> None:
        self.path = Path(path)
        self.rank = os.getpid() if rank is None else rank
        self.window_id = window_id
        self._span_id = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _next_span_id(self) -> int:
        self._span_id += 1
        return self._span_id

    def _write(self, payload: Dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    @contextmanager
    def span(self, name: str, **fields: Any) -> Iterator[None]:
        if name not in CANONICAL_SPANS:
            raise ValueError(f"sidecar span must be canonical, got {name!r}")
        span_id = self._next_span_id()
        start_ns = time.perf_counter_ns()
        try:
            yield
        finally:
            end_ns = time.perf_counter_ns()
            row = {
                "type": "span",
                "span_id": span_id,
                "name": name,
                "window_id": self.window_id,
                "start_ns": start_ns,
                "end_ns": end_ns,
                "rank": self.rank,
            }
            row.update(fields)
            self._write(row)

    def metric(self, name: str, value: float, **fields: Any) -> None:
        row = {
            "type": "metric",
            "name": name,
            "value": float(value),
            "rank": self.rank,
            "window_id": self.window_id,
        }
        row.update(fields)
        self._write(row)


class NullSpanRecorder:
    """Disabled producer with no serialization or filesystem work."""

    rank: Optional[int] = None
    path: Optional[Path] = None
    window_id: Optional[str] = None

    @contextmanager
    def span(self, name: str, **fields: Any) -> Iterator[None]:
        del name, fields
        yield

    def metric(self, name: str, value: float, **fields: Any) -> None:
        del name, value, fields
