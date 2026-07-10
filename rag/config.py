"""Load the base hyperparameters from config.toml into a frozen `CONFIG`.

The module-level constants in indexing/ and query/ read their values from here, so
retuning the system is a config edit, not a code change. Sections are flattened into
one dataclass (keys are unique across sections)."""

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    chunk_size: int
    chunk_overlap: int
    voyage_model: str
    method: str
    query_enhancement: str
    parent_document: bool
    hype: bool
    top_k: int
    relevance_threshold: float
    fuse_depth: int
    rrf_k: int
    vector_weight: float
    keyword_weight: float
    rerank_model: str
    rerank_depth: int
    source_window: int
    gen_provider: str
    gen_model: str
    gen_models: dict[str, str]
    answer_format: str
    max_tokens: int
    structured_max_tokens: int
    ocr_model: str
    agent_model: str
    research_subagent_model: str
    agent_top_k: int
    agent_method: str
    enable_hitl: bool
    research_budget: int


def _load() -> Config:
    with open(Path(__file__).parent / "config.toml", "rb") as f:
        sections = tomllib.load(f)
    flat = {key: value for section in sections.values() for key, value in section.items()}
    return Config(**flat)


CONFIG = _load()
