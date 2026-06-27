"""Split the loaded corpus into overlapping, metadata-rich chunks for embedding.

Run: uv run chunk.py

Reads documents via ingest.load_corpus, splits each into ~512-token chunks with
~50-token overlap, carries the document metadata + nearest heading onto every
chunk, and prints a sample spread across the corpus for hand inspection.

The splitting unit is the *line*, which is the right grain for this corpus:
for the Maia transcripts a line is one timestamped utterance, for the EPUB book
a line is one paragraph. We never split within a line, so chunk boundaries fall
between whole lines rather than mid-word.
"""

import os
from dataclasses import dataclass

from ingest import Document, count_tokens, load_corpus, CORPUS_ROOT

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "256"))  # 256 won the size sweep (chunking-experiments.md)
CHUNK_OVERLAP = 50  # tokens carried from the end of one chunk into the next


@dataclass
class Chunk:
    source: str          # document path; ties back to documents.source
    chunk_index: int     # position within the document
    content: str
    n_tokens: int
    title: str           # carried from the document
    section: str         # carried from the document
    date: str            # carried from the document
    heading: str | None  # nearest preceding heading line, if any
    # transcript-only metadata (None for books/PDFs/etc.):
    start: float | None = None               # earliest utterance time in chunk (s)
    end: float | None = None                 # latest utterance time (s)
    speakers: tuple[str, ...] | None = None  # distinct speakers present
    primary_speaker: str | None = None       # speaker with the most text in chunk


def looks_like_heading(line: str) -> bool:
    """A short, mostly-uppercase line — catches the book's PART/chapter headings.

    Heuristic. Real headings here ("PART ONE - THE ARCHITECTURE") are short and
    largely uppercase, while body prose almost never exceeds 60% uppercase. Two
    exclusions come straight from the inspection: transcript utterance lines
    start with a "[mm:ss]" timecode and are never headings, and the EPUB's
    drop-cap chapter initials land on their own line as a single letter.
    """
    line = line.strip()
    if line.startswith("["):
        return False
    words = line.split()
    letters = [c for c in line if c.isalpha()]
    if not (1 <= len(words) <= 20) or len(letters) < 4:
        return False
    uppercase_ratio = sum(c.isupper() for c in letters) / len(letters)
    return uppercase_ratio > 0.6


def _tail_overlap(units: list[tuple], overlap: int) -> tuple[list, int]:
    """Return the trailing units (text, n_tokens, ...) summing to about `overlap`."""
    kept: list[tuple] = []
    total = 0
    for unit in reversed(units):
        n = unit[1]
        if kept and total + n > overlap:
            break
        kept.append(unit)
        total += n
    kept.reverse()
    return kept, total


def _time_and_speakers(units: list[tuple]):
    """Chunk-level (start, end, speakers, primary_speaker) from its units.
    Speakers come from any labelled unit (transcripts and the dialogue PDF);
    start/end only from units that also carry timing (transcripts)."""
    chars: dict[str, int] = {}
    for text, _, _, _, speaker in units:
        if speaker:
            chars[speaker] = chars.get(speaker, 0) + len(text)
    speakers = tuple(sorted(chars)) or None
    primary = max(chars, key=chars.get) if chars else None
    timed = [(s, e) for _, _, s, e, _ in units if s is not None]
    start = min(s for s, _ in timed) if timed else None
    end = max(e for _, e in timed) if timed else None
    return start, end, speakers, primary


def chunk_document(
    doc: Document, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[Chunk]:
    """Greedily pack units into ~chunk_size-token chunks with ~overlap carryover.

    A unit is (text, n_tokens, start, end, speaker). Transcripts pack by timed
    utterance so each chunk carries its time range + speakers; other documents
    pack by text line, with start/end/speaker = None."""
    if doc.segments is not None:
        units = [(s.text, count_tokens(s.text), s.start, s.end, s.speaker)
                 for s in doc.segments]
    else:
        units = [(line, count_tokens(line), None, None, None)
                 for line in doc.text.split("\n") if line.strip()]

    chunks: list[Chunk] = []
    current: list[tuple] = []  # units in the chunk so far
    current_tokens = 0
    heading: str | None = None        # most recent heading seen
    chunk_heading: str | None = None  # heading in effect when this chunk began

    def flush() -> None:
        nonlocal current, current_tokens, chunk_heading
        content = "\n".join(text for text, *_ in current)
        start, end, speakers, primary = _time_and_speakers(current)
        chunks.append(
            Chunk(
                source=doc.source,
                chunk_index=len(chunks),
                content=content,
                n_tokens=current_tokens,
                title=doc.title,
                section=doc.section,
                date=doc.date,
                heading=chunk_heading,
                start=start,
                end=end,
                speakers=speakers,
                primary_speaker=primary,
            )
        )
        current, current_tokens = _tail_overlap(current, overlap)
        chunk_heading = heading

    for unit in units:
        text, n = unit[0], unit[1]
        if not current:
            chunk_heading = heading
        # A single unit longer than chunk_size becomes its own oversized chunk
        # rather than being split mid-line. Rare here; degrades gracefully.
        if current and current_tokens + n > chunk_size:
            flush()
        current.append(unit)
        current_tokens += n
        if looks_like_heading(text):
            heading = text

    if current:
        flush()
    return chunks


def chunk_corpus(documents: list[Document]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in documents:
        chunks.extend(chunk_document(doc))
    return chunks


# --- inspection --------------------------------------------------------------


def print_stats(chunks: list[Chunk]) -> None:
    token_counts = sorted(c.n_tokens for c in chunks)
    n = len(chunks)
    print("\n" + "=" * 70)
    print("CHUNK STATS")
    print("=" * 70)
    print(f"Chunks:          {n}")
    print(f"Tokens/chunk:    min {token_counts[0]}  median {token_counts[n // 2]}  max {token_counts[-1]}")
    print(f"With a heading:  {sum(c.heading is not None for c in chunks)}")

    print("\nChunks per document")
    for source in dict.fromkeys(c.source for c in chunks):
        doc_chunks = [c for c in chunks if c.source == source]
        print(f"  {len(doc_chunks):>4}  {source}")


def print_samples(chunks: list[Chunk], count: int = 20) -> None:
    """Print `count` chunks spread evenly across the corpus, head + tail of each,
    so boundaries (where mid-sentence splits would show up) are visible."""
    step = max(1, len(chunks) // count)
    print("\n" + "=" * 70)
    print(f"{count} SAMPLE CHUNKS (every {step}th)")
    print("=" * 70)
    for c in chunks[::step][:count]:
        print(f"\n[{c.section} / {c.title} #{c.chunk_index}]  {c.n_tokens} tokens"
              f"  heading={c.heading!r}")
        print("-" * 70)
        if len(c.content) <= 700:
            print(c.content)
        else:
            print(c.content[:400].rstrip())
            print("        … [middle elided] …")
            print(c.content[-250:].lstrip())


def main() -> None:
    documents = load_corpus(CORPUS_ROOT)
    chunks = chunk_corpus(documents)
    print_stats(chunks)
    print_samples(chunks)


if __name__ == "__main__":
    main()
