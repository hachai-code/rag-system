"""Schema for one row of rag_system_human_eval.jsonl, plus shared result aggregation."""

from collections import defaultdict
from typing import Literal

from pydantic import BaseModel


def pass_rate(rows: list[dict]) -> dict[str, float]:
    """Per-dimension pass rate over result rows carrying a `scores` dict."""
    per_dim = defaultdict(list)
    for r in rows:
        for dim, passed in r["scores"].items():
            per_dim[dim].append(passed)
    return {dim: sum(v) / len(v) for dim, v in sorted(per_dim.items())}


AxialCode = Literal[
    "A", "B", "C", "D", "E"
]  # A=Security/IP, B=Retrieval, C=Generation, D=Grounding, E=Formatting


class EvalItem(BaseModel):
    id: int
    question: str
    ideal_answer: str  # empty until filled in by hand
    axial_codes: list[AxialCode]  # first entry is the primary category used for the split
    difficulty: Literal["easy", "medium", "hard"]
    split: Literal["dev", "test"]
