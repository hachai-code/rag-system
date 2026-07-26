"""Shared helpers for eval scripts: JSONL loading and result aggregation."""

import json
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    """All rows of a JSONL file, skipping blank lines; [] if the file doesn't exist."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pass_rate(rows: list[dict]) -> dict[str, float]:
    """Per-dimension pass rate over result rows carrying a `scores` dict."""
    per_dim = defaultdict(list)
    for r in rows:
        for dim, passed in r["scores"].items():
            per_dim[dim].append(passed)
    return {dim: sum(v) / len(v) for dim, v in sorted(per_dim.items())}
