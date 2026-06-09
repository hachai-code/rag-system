"""Load the innerdance corpus into clean text + metadata, then print its shape.

Run: uv run ingest.py

This is the first stage of the RAG pipeline. It does NOT chunk or embed yet —
its only job is to turn a folder of raw files (RTF, PDF, EPUB, HTML, Markdown)
into a list of `Document`s with clean text, and to print enough stats that you
understand what you're working with before you start chunking.
"""

import re
import statistics
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import ebooklib
import tiktoken
from bs4 import BeautifulSoup
from ebooklib import epub
from pypdf import PdfReader
from striprtf.striprtf import rtf_to_text

CORPUS_ROOT = Path.home() / "Documents" / "innerdance corpus"

# tiktoken is OpenAI's tokenizer, not Claude's. We use it only for a fast,
# offline *estimate* of length so we can size chunks. Claude's real token
# counts will differ by a few percent — fine for understanding shape.
_encoder = tiktoken.get_encoding("o200k_base")


@dataclass
class Document:
    source: str    # path relative to the corpus root
    title: str     # human-readable name (from the filename)
    date: str      # ISO date the file was last modified
    section: str   # top-level corpus folder, e.g. "Maia" or "Documents"
    text: str      # cleaned plain text
    n_tokens: int  # estimated token count of `text`


# --- format-specific extractors: each returns raw (uncleaned) text -----------


def extract_rtf(path: Path) -> str:
    return rtf_to_text(path.read_text(encoding="utf-8", errors="ignore"))


def extract_pdf(path: Path) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() for page in reader.pages)


def extract_epub(path: Path) -> str:
    book = epub.read_epub(path)
    chapters = [
        item.get_content()
        for item in book.get_items()
        if item.get_type() == ebooklib.ITEM_DOCUMENT
    ]
    return "\n".join(html_to_text(html) for html in chapters)


def extract_html(path: Path) -> str:
    return html_to_text(path.read_bytes())


def extract_plain(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


EXTRACTORS = {
    ".rtf": extract_rtf,
    ".pdf": extract_pdf,
    ".epub": extract_epub,
    ".html": extract_html,
    ".htm": extract_html,
    ".md": extract_plain,
    ".txt": extract_plain,
}


# --- cleaning ----------------------------------------------------------------


def html_to_text(html: str | bytes) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator="\n")


def clean_text(text: str) -> str:
    """Strip per-line whitespace and collapse runs of blank lines."""
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --- loading -----------------------------------------------------------------


def load_document(path: Path, root: Path) -> Document:
    raw = EXTRACTORS[path.suffix.lower()](path)
    text = clean_text(raw)
    relative = path.relative_to(root)
    return Document(
        source=str(relative),
        title=path.stem,
        date=date.fromtimestamp(path.stat().st_mtime).isoformat(),
        section=relative.parts[0] if len(relative.parts) > 1 else "(root)",
        text=text,
        n_tokens=len(_encoder.encode(text)),
    )


def load_corpus(root: Path) -> list[Document]:
    documents = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in EXTRACTORS:
            continue
        print(f"  loading {path.relative_to(root)} ...", flush=True)
        documents.append(load_document(path, root))
    return documents


# --- reporting ---------------------------------------------------------------


def percentile(sorted_values: list[int], fraction: float) -> int:
    index = round(fraction * (len(sorted_values) - 1))
    return sorted_values[index]


def print_stats(documents: list[Document]) -> None:
    token_counts = sorted(doc.n_tokens for doc in documents)
    total_tokens = sum(token_counts)

    print("\n" + "=" * 70)
    print("CORPUS SHAPE")
    print("=" * 70)
    print(f"Documents:     {len(documents)}")
    print(f"Total tokens:  {total_tokens:,} (~{total_tokens / 1000:.0f}k)")
    print(f"Total chars:   {sum(len(d.text) for d in documents):,}")

    print("\nToken count distribution")
    print(f"  min     {token_counts[0]:>8,}")
    print(f"  p25     {percentile(token_counts, 0.25):>8,}")
    print(f"  median  {int(statistics.median(token_counts)):>8,}")
    print(f"  mean    {int(statistics.mean(token_counts)):>8,}")
    print(f"  p75     {percentile(token_counts, 0.75):>8,}")
    print(f"  max     {token_counts[-1]:>8,}")

    print("\nPer-document length (one bar = relative token count)")
    longest = max(token_counts)
    for doc in sorted(documents, key=lambda d: d.n_tokens, reverse=True):
        bar = "#" * round(40 * doc.n_tokens / longest)
        print(f"  {doc.n_tokens:>7,} {bar:<40} {doc.section}/{doc.title}")

    shortest_doc = min(documents, key=lambda d: d.n_tokens)
    longest_doc = max(documents, key=lambda d: d.n_tokens)
    print_preview("LONGEST", longest_doc)
    print_preview("SHORTEST", shortest_doc)


def print_preview(label: str, doc: Document) -> None:
    print("\n" + "-" * 70)
    print(f"{label}: {doc.title}  ({doc.n_tokens:,} tokens)")
    print(f"  source:  {doc.source}")
    print(f"  section: {doc.section}    date: {doc.date}")
    print("-" * 70)
    print(doc.text[:500].strip())
    print("  [...]")


def main() -> None:
    print(f"Loading corpus from: {CORPUS_ROOT}")
    documents = load_corpus(CORPUS_ROOT)
    print_stats(documents)


if __name__ == "__main__":
    main()
