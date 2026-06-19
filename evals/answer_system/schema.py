"""Schema for one row of rag_system_human_eval.jsonl."""

from typing import Literal

from pydantic import BaseModel

AxialCode = Literal["A", "B", "C", "D", "E"]  # A=Security/IP, B=Retrieval, C=Generation, D=Grounding, E=Formatting


class EvalItem(BaseModel):
    id: int
    question: str
    ideal_answer: str  # empty until filled in by hand
    axial_codes: list[AxialCode]  # first entry is the primary category used for the split
    difficulty: Literal["easy", "medium", "hard"]
    split: Literal["dev", "test"]
