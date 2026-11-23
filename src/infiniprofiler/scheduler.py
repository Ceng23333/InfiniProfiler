"""Retired InfiniLM scheduler-span helpers.

Promotion evidence is exclusive timing attribution. Optional free-form NVTX or
marker names may appear in provenance/warnings only; they are not a contract.
"""

from __future__ import annotations

from typing import Optional

# Kept as an empty frozenset so accidental imports fail closed.
CANONICAL_SPANS = frozenset()


def canonical_scheduler_family(name: str) -> Optional[str]:
    """Always None: canonical InfiniLM scheduler spans are no longer a contract."""
    del name
    return None


def scheduler_like(name: str) -> bool:
    """Deprecated; free-form marker names are never promotion evidence."""
    del name
    return False
