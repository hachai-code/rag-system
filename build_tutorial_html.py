"""Convert PYTHON_TUTORIAL.md into a styled, standalone HTML page.

A tiny, purpose-built Markdown converter — it only handles the constructs the
tutorial actually uses (headings, fenced code, inline code, bold, links, lists,
rules) so we need no third-party dependencies. Syntax highlighting is done by
highlight.js from a CDN, degrading to plain monospace if offline.

Run: uv run python build_tutorial_html.py  (or: python3 build_tutorial_html.py)
"""

import html
import re
from pathlib import Path

SRC = Path(__file__).parent / "PYTHON_TUTORIAL.md"
OUT = Path(__file__).parent / "python_tutorial.html"


def slug(text: str) -> str:
    """GitHub-style heading anchor: lowercase, drop punctuation, spaces -> '-'."""
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"\s+", "-", text.strip())


def inline(text: str) -> str:
    """Escape HTML, then apply inline code, bold, and links (in that order)."""
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
    return text


def convert(md: str) -> str:
    lines = md.split("\n")
    out: list[str] = []
    i = 0
    list_type: str | None = None  # "ul" | "ol" | None
    para: list[str] = []

    def close_para() -> None:
        if para:
            out.append(f"<p>{inline(' '.join(para))}</p>")
            para.clear()

    def close_list() -> None:
        nonlocal list_type
        if list_type:
            out.append(f"</{list_type}>")
            list_type = None

    while i < len(lines):
        line = lines[i]

        # Fenced code block: capture raw until the closing fence.
        fence = re.match(r"^```(\w*)\s*$", line)
        if fence:
            close_para()
            close_list()
            lang = fence.group(1) or "plaintext"
            body: list[str] = []
            i += 1
            while i < len(lines) and not re.match(r"^```\s*$", lines[i]):
                body.append(lines[i])
                i += 1
            i += 1  # skip closing fence
            code = html.escape("\n".join(body))
            out.append(
                f'<pre><code class="language-{lang}">{code}</code></pre>'
            )
            continue

        if not line.strip():
            close_para()
            close_list()
            i += 1
            continue

        if line.startswith("#"):
            close_para()
            close_list()
            level = len(line) - len(line.lstrip("#"))
            text = line[level:].strip()
            out.append(f'<h{level} id="{slug(text)}">{inline(text)}</h{level}>')
            i += 1
            continue

        if re.match(r"^---+\s*$", line):
            close_para()
            close_list()
            out.append("<hr>")
            i += 1
            continue

        ol = re.match(r"^\d+\.\s+(.*)", line)
        ul = re.match(r"^[-*]\s+(.*)", line)
        if ol or ul:
            close_para()
            want = "ol" if ol else "ul"
            if list_type != want:
                close_list()
                out.append(f"<{want}>")
                list_type = want
            item = (ol or ul).group(1)
            # fold continuation lines (indented, not blank, not a new item)
            while i + 1 < len(lines) and lines[i + 1].startswith("  ") \
                    and not re.match(r"^\s*([-*]|\d+\.)\s", lines[i + 1]):
                item += " " + lines[i + 1].strip()
                i += 1
            out.append(f"<li>{inline(item)}</li>")
            i += 1
            continue

        close_list()
        para.append(line.strip())
        i += 1

    close_para()
    close_list()
    return "\n".join(out)


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Learn Python by Reading a Real RAG System</title>
<link rel="stylesheet"
  href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/atom-one-dark.min.css">
<style>
  :root {{
    --bg: #fbfbfa; --fg: #24292f; --muted: #57606a; --line: #e3e3e0;
    --accent: #6f42c1; --code-bg: #282c34; --inline-bg: #efeef0;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: var(--bg); color: var(--fg);
    font: 17px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  }}
  .wrap {{ max-width: 860px; margin: 0 auto; padding: 4rem 1.5rem 6rem; }}
  h1 {{ font-size: 2.4rem; line-height: 1.15; margin: 0 0 .5rem; }}
  h2 {{
    font-size: 1.7rem; margin: 3.2rem 0 1rem; padding-top: 1rem;
    border-top: 1px solid var(--line);
  }}
  h3 {{ font-size: 1.25rem; margin: 2rem 0 .75rem; }}
  h2, h3 {{ scroll-margin-top: 1rem; }}
  a {{ color: var(--accent); text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  p {{ margin: 1rem 0; }}
  hr {{ border: 0; border-top: 1px solid var(--line); margin: 2.5rem 0; }}
  ul, ol {{ padding-left: 1.5rem; }}
  li {{ margin: .4rem 0; }}
  code {{
    font-family: "SF Mono", "JetBrains Mono", Menlo, Consolas, monospace;
    font-size: .88em;
  }}
  p code, li code, h2 code, h3 code {{
    background: var(--inline-bg); padding: .12em .38em; border-radius: 5px;
    color: #9a2cb3;
  }}
  pre {{
    background: var(--code-bg); border-radius: 10px; padding: 1.1rem 1.25rem;
    overflow-x: auto; margin: 1.2rem 0; font-size: .82rem; line-height: 1.55;
  }}
  pre code {{ background: none; padding: 0; color: #abb2bf; }}
  .lede {{ color: var(--muted); font-size: 1.1rem; }}
  .toc {{
    background: #fff; border: 1px solid var(--line); border-radius: 10px;
    padding: 1rem 1.25rem; margin: 2rem 0;
  }}
  .toc ol {{ margin: .25rem 0; }}
  footer {{
    margin-top: 4rem; padding-top: 1.5rem; border-top: 1px solid var(--line);
    color: var(--muted); font-size: .9rem;
  }}
</style>
</head>
<body>
<div class="wrap">
{body}
<footer>Generated from <code>PYTHON_TUTORIAL.md</code> in the rag-system repository.</footer>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<script>if (window.hljs) hljs.highlightAll();</script>
</body>
</html>
"""


def main() -> None:
    body = convert(SRC.read_text(encoding="utf-8"))
    OUT.write_text(PAGE.format(body=body), encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
