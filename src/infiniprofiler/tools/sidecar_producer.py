"""Minimal producer API for engines without HCTX instrumentation."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Union


class SidecarSpanRecorder:
    """Append explicit engine ranges to a JSONL file."""

    def __init__(self, path: Union[str, Path], rank: Optional[int] = None) -> None:
        self.path = Path(path)
        self.rank = os.getpid() if rank is None else rank
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, payload: Dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    @contextmanager
    def span(self, name: str, **fields: Any) -> Iterator[None]:
        start_ns = time.perf_counter_ns()
        try:
            yield
        finally:
            end_ns = time.perf_counter_ns()
            row = {
                "type": "span",
                "name": name,
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
        }
        row.update(fields)
        self._write(row)
