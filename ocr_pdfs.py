"""One-off: OCR the image-only slide PDFs into Markdown the pipeline can ingest.

Run: uv run ocr_pdfs.py  (needs ANTHROPIC_API_KEY)

Mandala.pdf and "Epilepsy and Brain Maps.pdf" are slideshow exports with their
content baked into page images, so pypdf extracts only stray titles. We render
each page with PyMuPDF and have Claude transcribe the slide (text + a one-line
note on any diagram) into Markdown, written next to the PDF as <stem>.md.
load_corpus then ingests the .md and skips the now-superseded .pdf.
"""

import base64
import os
from pathlib import Path

import anthropic
import fitz
from dotenv import load_dotenv

from ingest import CORPUS_ROOT
from rag import CLAUDE_MODEL

load_dotenv()

IMAGE_PDFS = ["Mandala.pdf", "Epilepsy and Brain Maps.pdf"]
MAX_EDGE = 1600  # px long edge; Claude downsizes vision images past ~1568px anyway, and this keeps each well under the 10 MB request cap

PROMPT = (
    "This is one slide from a slideshow. Transcribe its text exactly as Markdown, "
    "keeping headings, bullets, and emphasis. If the slide is mainly a diagram or "
    "chart, add a one-line italic description of what it shows including its labels "
    "and numbers. Output only the slide content, no preamble."
)


def transcribe_page(client: anthropic.Anthropic, png: bytes) -> str:
    b64 = base64.standard_b64encode(png).decode()
    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1500,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                {"type": "text", "text": PROMPT},
            ],
        }],
    )
    return resp.content[0].text.strip()


def ocr_pdf(client: anthropic.Anthropic, path: Path) -> str:
    doc = fitz.open(path)
    slides = []
    for i, page in enumerate(doc):
        zoom = MAX_EDGE / max(page.rect.width, page.rect.height)
        png = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom)).tobytes("png")
        text = transcribe_page(client, png)
        print(f"  page {i + 1}/{len(doc)} ({len(text)} chars)", flush=True)
        slides.append(f"## Slide {i + 1}\n\n{text}")
    return "\n\n".join(slides)


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set. Add it to .env or export it.")
    client = anthropic.Anthropic()
    for name in IMAGE_PDFS:
        pdf = CORPUS_ROOT / "Documents" / name
        print(f"OCR {name} with {CLAUDE_MODEL} ...")
        md = ocr_pdf(client, pdf)
        out = pdf.with_suffix(".md")
        out.write_text(f"# {pdf.stem}\n\n{md}\n")
        print(f"  wrote {out.relative_to(CORPUS_ROOT)} ({len(md)} chars)\n")


if __name__ == "__main__":
    main()
