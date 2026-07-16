"""Query stage 2: answer over the retrieved chunks, with grounded citations.

Two provider paths behind one `answer()` — Anthropic (native Citations API) and an
OpenAI-compatible path (OpenRouter via Pydantic AI structured output, citations rebuilt
from that output). See README "Generation / provider seam".
"""

from typing import TypedDict

import anthropic
from pydantic import BaseModel
from pydantic_ai import Agent, PromptedOutput
from pydantic_ai.settings import ModelSettings

from ..clients import openrouter_client, openrouter_model
from ..config import CONFIG
from ..db import Hit


class Citation(TypedDict):
    """One citation record tying a claim back to its chunk — the shape every adapter
    returns (app.py's Citation response model mirrors it)."""

    claim: str
    cited_text: str
    chunk_id: int
    title: str
    source: str


# Provider seam: which adapter answer() dispatches to and which model it runs.
# Prod runs the openai-compat/DeepSeek path.
GEN_PROVIDER = CONFIG.gen_provider
GEN_MODEL = CONFIG.gen_model
# Picker options for the API `model` field: short name -> OpenRouter model id.
GEN_MODELS = CONFIG.gen_models
# "prose" (one synthesized answer, chunks shown as proof) or "claims" (structured,
# per-sentence citations). Only affects the openai-compat path; anthropic is always prose.
ANSWER_FORMAT = CONFIG.answer_format

MAX_TOKENS = CONFIG.max_tokens
# The structured (JSON claims) response runs longer than raw prose (~2.5k tokens for a
# full answer), so the OpenAI-compatible path needs a higher ceiling.
STRUCTURED_MAX_TOKENS = CONFIG.structured_max_tokens

SYSTEM_PROMPT = (
    "You answer questions about the innerdance corpus using only the provided "
    "documents. If the documents do not contain the answer, say you don't know. "
    "Write a thorough, detailed answer in flowing prose that synthesizes and "
    "connects information across the documents rather than listing separate facts. "
    "Stay true to the source's own wording and meaning: prefer the documents' terms "
    "and phrasing, and do not add interpretations, inferences, or claims that the "
    "documents do not state. "
    "Begin with the substance of the answer itself. Never open with a meta-comment "
    'about your sources or method — do not start with phrases like "Based solely on '
    'the provided documents", "According to the documents", or "Based on the '
    'context"; go straight to the answer.'
)


def complete(prompt: str, model: str) -> str:
    """A bare OpenAI-compatible completion returning raw text — the lightweight
    counterpart to answer() for auxiliary generation (HyDE hypotheticals, multi-query
    paraphrases) that needs no citation scaffolding. Runs the same OpenRouter seam."""
    client = openrouter_client()
    resp = client.chat.completions.create(
        model=model,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


def context_documents(hits: list[Hit]) -> list[dict]:
    """Each retrieved chunk as a citable document block. Passing chunks as separate
    documents is what lets the Citations API map `document_index` back to hits[index]."""
    return [
        {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": hit["content"]},
            "title": hit["title"],
            "citations": {"enabled": True},
        }
        for hit in hits
    ]


def _messages(question: str, hits: list[Hit]) -> list[dict]:
    """The user turn: the retrieved chunks as citable documents, then the question."""
    return [
        {
            "role": "user",
            "content": context_documents(hits)
            + [{"type": "text", "text": f"Question: {question}"}],
        }
    ]


def _openai_user_content(question: str, hits: list[Hit], numbered: bool = True) -> str:
    """The OpenAI-compatible user turn: the chunks, then the question.

    The claims path numbers each chunk [i] so the model can cite it by index (mapping
    back to hits[i]). Prose passes numbered=False: with no [i] markers in the input, the
    model has nothing to echo, so its prose stays free of stray citation brackets."""
    if numbered:
        chunks = "\n\n".join(f"[{i}] {hit['content']}" for i, hit in enumerate(hits))
    else:
        chunks = "\n\n".join(hit["content"] for hit in hits)
    return f"{chunks}\n\nQuestion: {question}"


def _citations(content: list, hits: list[Hit]) -> list[Citation]:
    """One record per citation, tying each cited claim back to its chunk.

    `cited_text` is extracted by the API from the document, so it can't be a quote the
    source doesn't contain."""
    return [
        {
            "claim": block.text,
            "cited_text": c.cited_text,
            "chunk_id": hits[c.document_index]["id"],
            "title": hits[c.document_index]["title"],
            "source": hits[c.document_index]["source"],
        }
        for block in content
        for c in (block.citations or [])
    ]


def _proof(hits: list[Hit]) -> list[Citation]:
    """The retrieved chunks shown as-is as proof for a prose answer — no claim/span
    mapping. Same dict shape as _citations()/_chunk_citations() so callers (app.py's
    Citation model) don't care which path produced them."""
    return [
        {
            "claim": "",
            "cited_text": hit["content"],
            "chunk_id": hit["id"],
            "title": hit["title"],
            "source": hit["source"],
        }
        for hit in hits
    ]


def answer_prose(
    question: str, hits: list[Hit], model: str = GEN_MODEL, system: str = SYSTEM_PROMPT
) -> tuple[str, list[Citation]]:
    """One synthesized prose answer over the chunks (openai-compat path).

    No structured schema — a raw completion is what makes the model write flowing
    prose instead of the per-sentence claims of the GroundedAnswer path. The proof is
    the retrieved chunks; grounding is by the prompt, not by construction."""
    client = openrouter_client()
    # ponytail: STRUCTURED_MAX_TOKENS ceiling; if OpenRouter truncates a long answer
    # the text just comes back short (finish_reason="length"), raise the cap if it bites.
    resp = client.chat.completions.create(
        model=model,
        max_tokens=STRUCTURED_MAX_TOKENS,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": _openai_user_content(question, hits, numbered=False)},
        ],
    )
    return resp.choices[0].message.content, _proof(hits)


class Claim(BaseModel):
    """One sentence of the answer plus the [i] indices of the chunks supporting it."""

    statement: str
    chunk_indices: list[int]


class GroundedAnswer(BaseModel):
    claims: list[Claim]


def _chunk_citations(grounded: GroundedAnswer, hits: list[Hit]) -> tuple[str, list[Citation]]:
    """Assemble (answer_text, citations) from claims tagged with chunk indices.

    Each citation's `cited_text` is the chunk's own content — same dict shape as
    `_citations()`, so callers don't care which adapter produced it, and grounding
    holds by construction. Out-of-range indices are dropped."""
    text = " ".join(claim.statement for claim in grounded.claims)
    citations = [
        {
            "claim": claim.statement,
            "cited_text": hits[idx]["content"],
            "chunk_id": hits[idx]["id"],
            "title": hits[idx]["title"],
            "source": hits[idx]["source"],
        }
        for claim in grounded.claims
        for idx in claim.chunk_indices
        if 0 <= idx < len(hits)
    ]
    return text, citations


def answer(
    question: str,
    hits: list[Hit],
    model: str = GEN_MODEL,
    system: str = SYSTEM_PROMPT,
    provider: str = GEN_PROVIDER,
    fmt: str = ANSWER_FORMAT,
) -> tuple[str, list[Citation]]:
    """Answer over the retrieved chunks and return (answer_text, citations).

    Dispatches by `provider`; `fmt` picks prose vs structured claims on
    the openai-compat path. The eval runner overrides the defaults to test a config
    (evals/run.py)."""
    if provider == "anthropic":
        # Already prose with native span citations, so `fmt` is a no-op here.
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=_messages(question, hits),
        )
        text = "".join(block.text for block in response.content)
        return text, _citations(response.content, hits)

    if fmt == "prose":
        return answer_prose(question, hits, model=model, system=system)

    # PromptedOutput, not tool calls: some OpenRouter models don't reliably emit tool
    # calls for the schema, so we ask for the GroundedAnswer as JSON in the content.
    # output_retries re-asks the model when the JSON doesn't parse — OpenRouter
    # occasionally truncates a long answer under load and succeeds on retry.
    agent = Agent(
        openrouter_model(model),
        output_type=PromptedOutput(GroundedAnswer),
        instructions=system,
        model_settings=ModelSettings(max_tokens=STRUCTURED_MAX_TOKENS),
        output_retries=2,
    )
    result = agent.run_sync(_openai_user_content(question, hits))
    return _chunk_citations(result.output, hits)


def answer_stream(
    question: str,
    hits: list[Hit],
    model: str = GEN_MODEL,
    provider: str = GEN_PROVIDER,
    fmt: str = ANSWER_FORMAT,
    system: str = SYSTEM_PROMPT,
):
    """Yield the answer incrementally, then one citation record per source.

    Both adapters keep the text-first / citations-last contract. Prose token-streams
    (Anthropic natively, the openai-compat path via a raw streamed completion); the
    structured claims format must be parsed whole, so it comes back as one text event.
    Each item is {"type": "text"|"citation", ...}."""
    if provider == "anthropic":
        client = anthropic.Anthropic()
        with client.messages.stream(
            model=model,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=_messages(question, hits),
        ) as stream:
            for text in stream.text_stream:
                yield {"type": "text", "text": text}
            final = stream.get_final_message()
        for cite in _citations(final.content, hits):
            yield {"type": "citation", **cite}
        return

    if fmt == "claims":
        text, citations = answer(
            question, hits, model=model, provider=provider, fmt=fmt, system=system
        )
        yield {"type": "text", "text": text}
        for cite in citations:
            yield {"type": "citation", **cite}
        return

    # Prose: token-stream the raw completion, then the retrieved chunks as proof.
    client = openrouter_client()
    stream = client.chat.completions.create(
        model=model,
        max_tokens=STRUCTURED_MAX_TOKENS,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": _openai_user_content(question, hits, numbered=False)},
        ],
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield {"type": "text", "text": delta}
    for cite in _proof(hits):
        yield {"type": "citation", **cite}


if __name__ == "__main__":
    from rag.db import connect
    from rag.query.retrieve import rerank_search

    question = "What is the relationship between epilepsy and spiritual experience?"
    with connect() as conn:
        hits = rerank_search(conn, question)

    text, citations = answer(question, hits)
    print(f"Q: {question}\n")
    print(text)

    by_id = {hit["id"]: hit["content"] for hit in hits}
    print(f"\nCitations ({len(citations)}):")
    for cite in citations:
        # Groundedness check: cited_text is extracted from the chunk, so this should
        # always hold — if it fails, the citation is not real.
        grounded = cite["cited_text"] in by_id[cite["chunk_id"]]
        mark = "ok" if grounded else "HALLUCINATED"
        print(f'  [{mark}] {cite["title"][:40]} — "{cite["cited_text"][:70]}"')
