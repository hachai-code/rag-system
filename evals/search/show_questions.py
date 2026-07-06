"""Print synthetic_questions.jsonl in a human-readable form.

The raw JSONL is hard to skim because each row is one long line with the full
chunk text in `ideal_answer`. This lays each row out as question + source + a
trimmed preview of the ideal answer.

Run: uv run evals/search/show_questions.py [path]
"""

import json
import sys
import textwrap
from pathlib import Path

DEFAULT = Path(__file__).parent / "data" / "synthetic_questions.jsonl"


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ideal = " ".join(row["ideal_answer"].split())  # collapse transcript newlines
        print(f"Q{row['id']:>2}  ·  {row['source']['title'][:45]}  ·  chunk {row['expected_chunk_ids'][0]}")
        print(textwrap.fill(row["question"], width=90, initial_indent="    ", subsequent_indent="    "))
        print(textwrap.fill(f"ideal: {ideal[:220]}…", width=90, initial_indent="    ", subsequent_indent="    "))
        print()


if __name__ == "__main__":
    main()
