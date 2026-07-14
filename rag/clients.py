"""Shared OpenRouter seam: the base URL and a configured OpenAI client.

Langfuse-traced callers build their own langfuse.openai client and share
only the URL.
"""

import os

from openai import OpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def openrouter_client() -> OpenAI:
    """A plain OpenAI client pointed at OpenRouter."""
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=os.environ["OPENROUTER_API_KEY"])
