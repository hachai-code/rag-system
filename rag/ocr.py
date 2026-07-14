"""One-off: OCR the image-only slide PDFs into Markdown the pipeline can ingest.

Mandala.pdf and "Epilepsy and Brain Maps.pdf" are slideshow exports whose content is
baked into page images. We render each page and have a vision model transcribe it into
Markdown, written next to the PDF as <stem>.md (which load_corpus then prefers).
Run: uv run python -m rag.ocr  (needs OPENROUTER_API_KEY)
"""

import base64
import os
from pathlib import Path

import fitz
from openai import OpenAI

from .clients import openrouter_client
from .config import CONFIG
from .indexing.ingest import CORPUS_ROOT

IMAGE_PDFS = ["Mandala.pdf", "Epilepsy and Brain Maps.pdf"]
MAX_EDGE = 1600  # px long edge; keeps each page image well under model input limits

PROMPT = (
    "This is one slide from a slideshow. Transcribe its text exactly as Markdown, "
    "keeping headings, bullets, and emphasis. If the slide is mainly a diagram or "
    "chart, add a one-line italic description of what it shows including its labels "
    "and numbers. Output only the slide content, no preamble."
)


def transcribe_page(client: OpenAI, png: bytes) -> str:
    b64 = base64.standard_b64encode(png).decode()
    resp = client.chat.completions.create(
        model=CONFIG.ocr_model,
        max_tokens=1500,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": PROMPT},
            ],
        }],
    )
    return resp.choices[0].message.content.strip()


def ocr_pdf(client: OpenAI, path: Path) -> str:
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
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY is not set. Add it to .env or export it.")
    client = openrouter_client()
    for name in IMAGE_PDFS:
        pdf = CORPUS_ROOT / "Documents" / name
        print(f"OCR {name} with {CONFIG.ocr_model} ...")
        md = ocr_pdf(client, pdf)
        out = pdf.with_suffix(".md")
        out.write_text(f"# {pdf.stem}\n\n{md}\n")
        print(f"  wrote {out.relative_to(CORPUS_ROOT)} ({len(md)} chars)\n")


if __name__ == "__main__":
    main()
