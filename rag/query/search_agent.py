"""The naked agent loop: a while loop in which the model either calls the one tool
(corpus search) or returns a final answer; we execute the tool, append the result,
and repeat. The agentic counterpart to the fixed retrieve-then-answer pipeline in
answer.py — no framework, same OpenRouter seam as complete()."""

import json
import os

import psycopg
from openai import OpenAI

from ..config import CONFIG
from .retrieve import search

MODEL = CONFIG.gen_model
MAX_TOKENS = CONFIG.max_tokens
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MAX_ITERATIONS = 5

SYSTEM_PROMPT = (
    "You answer questions about the innerdance corpus. Use search_corpus to find "
    "relevant material before answering; search again with a different query if the "
    "results don't cover the question. If the corpus does not contain the answer, "
    "say you don't know."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_corpus",
            "description": "Search the innerdance corpus. Returns the most relevant chunks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                },
                "required": ["query"],
            },
        },
    }
]


def run_agent(conn: psycopg.Connection, question: str) -> str:
    """Answer the question by letting the model drive retrieval: it decides when to
    search and when it has enough to answer."""
    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=os.environ["OPENROUTER_API_KEY"])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    iterations = 0
    while iterations < MAX_ITERATIONS:
        iterations += 1
        msg = client.chat.completions.create(
            model=MODEL, max_tokens=MAX_TOKENS, tools=TOOLS, messages=messages
        ).choices[0].message
        messages.append(msg)
        if not msg.tool_calls:
            return msg.content
        for call in msg.tool_calls:
            query = json.loads(call.function.arguments)["query"]
            print(f"-> search_corpus({query!r})")
            hits = search(conn, query, k=5)
            result = "\n\n".join(f"[{hit['title']}]\n{hit['content']}" for hit in hits)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
    return "Reached max iterations without a final answer."


if __name__ == "__main__":
    import sys

    from rag.db import connect

    question = sys.argv[1] if len(sys.argv) > 1 else "What is innerdance?"
    with connect() as conn:
        text = run_agent(conn, question)
    print(f"\nQ: {question}\n")
    print(text)
