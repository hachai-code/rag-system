"""Stage 1: load the corpus into clean `Document`s. See README "Pipeline order".

Run `uv run python -m rag.indexing.ingest` to print the corpus shape.
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

# tiktoken is OpenAI's tokenizer; used only for a fast offline length estimate.
_encoder = tiktoken.get_encoding("o200k_base")


def count_tokens(text: str) -> int:
    return len(_encoder.encode(text))


@dataclass
class Segment:
    """One unit of attributed text: a timed ASR utterance, or a sentence of a
    speaker's turn in the dialogue PDF (which carries no timing)."""

    start: float | None
    end: float | None
    speaker: str | None  # "pi" | "participant" | "doc romy" | None
    text: str  # speaker label kept inline ("pi: ...")


@dataclass
class Document:
    source: str  # path relative to the corpus root
    title: str
    date: str  # ISO date the file was last modified
    section: str  # top-level corpus folder, e.g. "Maia"
    text: str
    n_tokens: int
    segments: list[Segment] | None = None  # timed utterances, transcripts only


# --- format-specific extractors: each returns raw (uncleaned) text -----------


def extract_rtf(path: Path) -> str:
    return rtf_to_text(path.read_text(encoding="utf-8", errors="ignore"))


def extract_pdf(path: Path) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() for page in reader.pages)


def extract_epub(path: Path) -> str:
    book = epub.read_epub(path)
    chapters = [
        item.get_content() for item in book.get_items() if item.get_type() == ebooklib.ITEM_DOCUMENT
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


# ASR timecode: "[00:01.000 --> 00:03.500]", optionally with an hour component.
TIMECODE = re.compile(r"\[\d{1,2}:\d{2}(?::\d{2})?\.\d+ --> \d{1,2}:\d{2}(?::\d{2})?\.\d+\]")


def is_transcript(text: str) -> bool:
    """True for the Maia transcripts, which open with a timecode."""
    return bool(TIMECODE.match(text.lstrip()))


def reflow_transcript(text: str) -> str:
    """Turn timecoded ASR utterances into flowing, sentence-split prose.

    The timecodes are ~40% of the tokens and fragment every sentence, which muddies
    the embeddings; see README "Pipeline order"."""
    utterances = []
    for line in text.splitlines():
        utterance = TIMECODE.sub("", line).strip()
        if utterance:
            utterances.append(utterance)
    flowing = " ".join(utterances)
    sentences = re.split(r"(?<=[.!?]) +", flowing)
    return "\n".join(sentences)


# One transcript line: "[start --> end] speaker: words".
SEGMENT_LINE = re.compile(r"^\[(\S+) --> (\S+)\]\s*(.*)$")
SPEAKER = re.compile(r"^(pi|participant):")


def _to_seconds(stamp: str) -> float:
    """'12:46.847' (mm:ss.s) or '1:22:53.218' (h:mm:ss.s) -> seconds."""
    parts = stamp.split(":")
    seconds = float(parts[-1]) + int(parts[-2]) * 60
    if len(parts) == 3:
        seconds += int(parts[0]) * 3600
    return seconds


def parse_transcript(text: str) -> list[Segment]:
    """Pull start/end/speaker out of each timecoded line for chunk metadata. The
    speaker label stays in the text, so the embedded content is unchanged."""
    segments = []
    for line in text.splitlines():
        m = SEGMENT_LINE.match(line.strip())
        if not m:
            continue
        body = m.group(3).strip()
        spk = SPEAKER.match(body)
        segments.append(
            Segment(
                _to_seconds(m.group(1)),
                _to_seconds(m.group(2)),
                spk.group(1) if spk else None,
                body,
            )
        )
    return segments


# --- dialogue PDF ------------------------------------------------------------

# transformation_medicine_ebook.pdf is a two-voice dialogue: Pi's lines are italic
# (Delicious-Italic), Doc Romy's roman (Delicious-Roman). Plain-text extraction loses
# the font, so we read per-span fonts to recover who's speaking.
DIALOGUE_PDF = "transformation_medicine_ebook.pdf"
_PAGE_NUMBER = re.compile(
    r"\d[\d\s]*\|[\d\s]*"
)  # footer page numbers leak as italic spans: "12 | 13"
_FRONT_MATTER = re.compile(r"ISBN|Copyright|Published by", re.IGNORECASE)


def extract_dialogue_pdf(path: Path) -> tuple[str, list[Segment]]:
    """Recover Pi (italic) / Doc Romy (roman) turns from the dialogue ebook, then
    split each turn into sentences — the grain chunk.py packs by."""
    reader = PdfReader(path)
    spans: list[tuple[str, str]] = []

    def visit(text, cm, tm, fontdict, size):
        font = (fontdict or {}).get("/BaseFont", "")
        if "Delicious-Italic" in font:
            speaker = "pi"
        elif "Delicious-Roman" in font:
            speaker = "doc romy"
        else:
            return  # bold headings + Perpetua running headers/page numbers
        cleaned = _PAGE_NUMBER.sub(" ", text)
        if cleaned.strip():
            spans.append((speaker, cleaned))

    for page in reader.pages:
        page.extract_text(visitor_text=visit)

    turns: list[tuple[str, str]] = []
    for speaker, text in spans:
        if turns and turns[-1][0] == speaker:
            turns[-1] = (speaker, turns[-1][1] + text)
        else:
            turns.append((speaker, text))

    # Drop title/copyright front matter: start at the first substantial,
    # non-boilerplate turn (the "My name is Romy" self-introduction).
    start = next(
        (
            i
            for i, (_, t) in enumerate(turns)
            if len(t.strip()) > 300 and not _FRONT_MATTER.search(t)
        ),
        0,
    )

    segments: list[Segment] = []
    for speaker, text in turns[start:]:
        flowing = " ".join(text.split())
        for i, sentence in enumerate(re.split(r"(?<=[.!?]) +", flowing)):
            if sentence.strip():
                label = f"{speaker}: " if i == 0 else ""
                segments.append(Segment(None, None, speaker, label + sentence))
    return "\n".join(s.text for s in segments), segments


# --- loading -----------------------------------------------------------------


def load_document(path: Path, root: Path) -> Document:
    if path.name == DIALOGUE_PDF:
        text, segments = extract_dialogue_pdf(path)
    else:
        raw = EXTRACTORS[path.suffix.lower()](path)
        segments = None
        if is_transcript(raw):
            segments = parse_transcript(raw)
            raw = reflow_transcript(raw)
        text = clean_text(raw)
    relative = path.relative_to(root)
    return Document(
        source=str(relative),
        title=path.stem.removesuffix(".timestamped"),  # "Foo.timestamped.txt" -> "Foo"
        date=date.fromtimestamp(path.stat().st_mtime).isoformat(),
        section=relative.parts[0] if len(relative.parts) > 1 else "(root)",
        text=text,
        n_tokens=count_tokens(text),
        segments=segments,
    )


def load_corpus(root: Path) -> list[Document]:
    documents = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in EXTRACTORS:
            continue
        if path.suffix.lower() == ".pdf" and path.with_suffix(".md").exists():
            continue  # OCR'd Markdown sidecar supersedes the image-only PDF
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
