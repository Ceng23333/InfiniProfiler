"""Canonical InfiniLM scheduler-span names and families."""

from __future__ import annotations

import re
from typing import Optional


QUEUE_WAIT = "infinilm:queue_wait"
ADMISSION = "infinilm:admission"
BATCH_BUILD = "infinilm:batch_build"
SCHEDULE = "infinilm:schedule"
FORWARD_STEP = "infinilm:forward_step"
SAMPLE = "infinilm:sample"

CANONICAL_SPANS = frozenset(
    {
        QUEUE_WAIT,
        ADMISSION,
        BATCH_BUILD,
        SCHEDULE,
        FORWARD_STEP,
        SAMPLE,
    }
)

SCHEDULER_LIKE_RE = re.compile(
    r"(scheduler|schedule|batch|step|forward|sample|admission|queue)", re.IGNORECASE
)


def canonical_scheduler_family(name: str) -> Optional[str]:
    """Return the stable scheduler family for an exact canonical span name."""
    if name not in CANONICAL_SPANS:
        return None
    if name in (QUEUE_WAIT, ADMISSION):
        return "queue_admission"
    if name == BATCH_BUILD:
        return "batch_packing"
    return "step_control"


def scheduler_like(name: str) -> bool:
    """Identify names that look scheduler-related but are not canonical facts."""
    return SCHEDULER_LIKE_RE.search(name) is not None
