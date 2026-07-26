"""Shared API-client seam: OpenRouter base URL and lazily built clients.

Clients are built on first use, not at import, so importing rag needs no API
keys. Langfuse-traced callers build their own langfuse.openai client and share
only the URL.
"""

import os
from functools import lru_cache

import voyageai
from openai import OpenAI
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def openrouter_client() -> OpenAI:
    """A plain OpenAI client pointed at OpenRouter."""
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=os.environ["OPENROUTER_API_KEY"])


@lru_cache(maxsize=8)
def openrouter_model(model: str) -> OpenAIChatModel:
    """The Pydantic AI model over the OpenRouter seam, for the structured-output path in
    answer.py. Cached per model id (mirrors voyage_client); reads OPENROUTER_API_KEY on
    first use so importing rag needs no keys."""
    return OpenAIChatModel(
        model, provider=OpenRouterProvider(api_key=os.environ["OPENROUTER_API_KEY"])
    )


@lru_cache(maxsize=1)
def voyage_client() -> voyageai.Client:
    """The shared Voyage client (embeddings + rerank). Reads VOYAGE_API_KEY on first use."""
    return voyageai.Client(max_retries=2)
