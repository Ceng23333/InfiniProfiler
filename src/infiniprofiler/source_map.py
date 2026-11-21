"""Configurable source discovery for promotion-grade graph attribution.

The scanner is engine-neutral. InfiniLM-specific symbols are supplied as rules
or required symbols by the campaign, never hard-coded into the core tool.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


SOURCE_MAP_SCHEMA = "source-map-1"
SOURCE_MAP_RULES_SCHEMA = "source-map-rules-1"
TEXT_SUFFIXES = {
    ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".cu", ".cuh",
    ".py", ".rs", ".toml", ".yaml", ".yml", ".json", ".sh", ".bash",
}
SKIP_DIRECTORIES = {
    ".git", ".hg", ".svn", "__pycache__", "node_modules",
    "build", "dist", "target", ".venv", "venv",
}


class SourceMapError(ValueError):
    """Raised when source-map rules or the source tree cannot be scanned."""


@dataclass(frozen=True)
class SourceMapRule:
    kind: str
    patterns: Sequence[str]
    provider: Optional[str] = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SourceMapRule":
        kind = raw.get("kind")
        patterns = raw.get("patterns")
        provider = raw.get("provider")
        if kind not in ("graph_guard", "provider", "scheduler", "deployment_entrypoint", "symbol"):
            raise SourceMapError(f"source-map rule kind is invalid: {kind!r}")
        if not isinstance(patterns, list) or not patterns or any(not isinstance(item, str) or not item for item in patterns):
            raise SourceMapError("source-map rule patterns must be a non-empty string list")
        if kind == "provider" and (not isinstance(provider, str) or not provider):
            raise SourceMapError("provider rule requires a provider name")
        for pattern in patterns:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise SourceMapError(f"invalid source-map pattern {pattern!r}: {exc}") from exc
        return cls(kind=str(kind), patterns=tuple(patterns), provider=None if provider is None else str(provider))


def load_rules(path: Path) -> List[SourceMapRule]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceMapError(f"cannot read source-map rules {path}: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != SOURCE_MAP_RULES_SCHEMA:
        raise SourceMapError(f"source-map rules schema_version must be {SOURCE_MAP_RULES_SCHEMA}")
    raw_rules = raw.get("rules")
    if not isinstance(raw_rules, list):
        raise SourceMapError("source-map rules must be an array")
    return [SourceMapRule.from_dict(rule) for rule in raw_rules]


def default_rules() -> List[SourceMapRule]:
    """Return generic rules; campaign-specific symbols are supplied separately."""
    return [
        SourceMapRule(
            kind="graph_guard",
            patterns=(r"supports_.*graph_capture", r"graph_capture_eligible", r"graph_guard"),
        ),
        SourceMapRule(
            kind="provider",
            provider="flash_attention",
            patterns=(r"flash_attn|flash_attention",),
        ),
        SourceMapRule(
            kind="provider",
            provider="native_attention",
            patterns=(r"native_attention|naive_attention",),
        ),
        SourceMapRule(
            kind="scheduler",
            patterns=(r"schedule_batch|scheduler_step|run_scheduler|admission_policy",),
        ),
        SourceMapRule(
            kind="deployment_entrypoint",
            patterns=(r"api_server|entrypoint|serve\(", r"if __name__ == .__main__.:",),
        ),
    ]


def _iter_source_files(source_root: Path) -> Iterable[Path]:
    for path in sorted(source_root.rglob("*")):
        if not path.is_file():
            continue
        relative_parts = path.relative_to(source_root).parts
        if any(part in SKIP_DIRECTORIES for part in relative_parts):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        yield path


def _source_root_hash(files: Sequence[Path], source_root: Path) -> str:
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(source_root).as_posix()
        data = path.read_bytes()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _line_matches(line: str, rule: SourceMapRule) -> Optional[str]:
    for pattern in rule.patterns:
        if re.search(pattern, line):
            return pattern
    return None


def build_source_map(
    source_root: Path,
    *,
    source_pin: str,
    rules: Optional[Sequence[SourceMapRule]] = None,
    required_symbols: Sequence[str] = (),
) -> Dict[str, Any]:
    """Scan a source tree and return a source-map-1 document.

    Required symbols are campaign inputs. They are recorded by exact literal
    occurrence so a graph lever can cite the file and line that was inspected.
    """
    if not source_pin:
        raise SourceMapError("source_pin is required")
    root = source_root.expanduser().resolve()
    if not root.is_dir():
        raise SourceMapError(f"source root is not a directory: {root}")
    active_rules = list(rules) if rules is not None else default_rules()
    required = list(dict.fromkeys(item for item in required_symbols if item))
    if not required:
        raise SourceMapError("at least one required symbol is required")

    files = list(_iter_source_files(root))
    symbols: List[Dict[str, Any]] = []
    provider_matches: Dict[str, Dict[str, Any]] = {}
    artifacts: Dict[str, Dict[str, Any]] = {}

    for path in files:
        relative = path.relative_to(root).as_posix()
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            raise SourceMapError(f"cannot read {path}: {exc}") from exc
        for line_number, line in enumerate(lines, 1):
            stripped = line.strip()
            for symbol in required:
                # Exact source hits must be code; a matching comment would not
                # be sufficient provenance for a lever contract.
                if symbol in line and not stripped.startswith(("#", "//")):
                    if any(
                        symbol_entry.get("symbol") == symbol
                        and symbol_entry.get("file") == relative
                        and symbol_entry.get("line_start") == line_number
                        for symbol_entry in symbols
                    ):
                        continue
                    lowered = symbol.lower()
                    kind = (
                        "graph_guard"
                        if ("graph" in lowered and ("capture" in lowered or "guard" in lowered))
                        else "symbol"
                    )
                    symbols.append(
                        {
                            "symbol": symbol,
                            "file": relative,
                            "line_start": line_number,
                            "kind": kind,
                            "match": stripped[:500],
                        }
                    )
            for rule in active_rules:
                pattern = _line_matches(line, rule)
                if pattern is None:
                    continue
                match = {
                    "file": relative,
                    "line_start": line_number,
                    "pattern": pattern,
                    "match": stripped[:500],
                }
                if rule.kind == "provider":
                    provider_name = rule.provider or "unknown"
                    provider = provider_matches.setdefault(
                        provider_name,
                        {"provider": provider_name, "patterns": [], "files": [], "matches": []},
                    )
                    if pattern not in provider["patterns"]:
                        provider["patterns"].append(pattern)
                    if relative not in provider["files"]:
                        provider["files"].append(relative)
                    provider["matches"].append(match)
                else:
                    artifact = artifacts.setdefault(
                        rule.kind,
                        {"kind": rule.kind, "patterns": [], "files": [], "matches": []},
                    )
                    if pattern not in artifact["patterns"]:
                        artifact["patterns"].append(pattern)
                    if relative not in artifact["files"]:
                        artifact["files"].append(relative)
                    artifact["matches"].append(match)

    missing = [
        symbol
        for symbol in required
        if not any(entry.get("symbol") == symbol for entry in symbols)
    ]
    if missing:
        raise SourceMapError(f"required symbols not found: {', '.join(missing)}")

    return {
        "schema_version": SOURCE_MAP_SCHEMA,
        "source_pin": source_pin,
        "source_root": str(root),
        "source_root_hash": _source_root_hash(files, root),
        "scanned_file_count": len(files),
        "symbols": sorted(symbols, key=lambda item: (item["file"], item["line_start"], item["symbol"])),
        "providers": sorted(provider_matches.values(), key=lambda item: item["provider"]),
        "artifacts": sorted(artifacts.values(), key=lambda item: item["kind"]),
        "rules": [
            {
                "kind": rule.kind,
                "patterns": list(rule.patterns),
                **({"provider": rule.provider} if rule.provider else {}),
            }
            for rule in active_rules
        ],
    }


def write_source_map(document: Mapping[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
