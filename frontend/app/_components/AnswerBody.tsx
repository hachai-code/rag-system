import type { Citation } from "@/lib/types";

// Render the answer with **bold** and inline citation markers placed right after
// the claim each citation backs. Each claim is an exact, in-order span of the
// answer (Anthropic citations), so we locate them sequentially; citations sharing
// a span cluster into one [n][m] group. The marker number matches the chip below.
export function AnswerBody({
  answer,
  citations,
  openSource,
}: {
  answer: string;
  citations: Citation[];
  openSource: (i: number, chunkId: number) => void;
}) {
  // Map each span's end offset to the citation indices that end there.
  const byEnd = new Map<number, number[]>();
  let cursor = 0;
  citations.forEach((c, i) => {
    const at = c.claim ? answer.indexOf(c.claim, cursor) : -1;
    if (at === -1) return; // claim not found in the text; it still shows in the chip list
    const end = at + c.claim.length;
    (byEnd.get(end) ?? byEnd.set(end, []).get(end)!).push(i);
    cursor = at; // ponytail: keep start, so a span cited by two sources matches both
  });

  const stops = [...byEnd.keys()].sort((a, b) => a - b);
  const nodes: React.ReactNode[] = [];
  let prev = 0;
  for (const stop of stops) {
    nodes.push(...renderBold(answer.slice(prev, stop), prev));
    for (const i of byEnd.get(stop)!) {
      nodes.push(
        <button
          key={`m${i}`}
          onClick={() => openSource(i, citations[i].chunk_id)}
          className="align-super text-xs text-blue-600 hover:underline"
        >
          [{i + 1}]
        </button>,
      );
    }
    prev = stop;
  }
  nodes.push(...renderBold(answer.slice(prev), prev));
  return <>{nodes}</>;
}

// Split a text run into plain strings and <strong> for **bold** spans.
function renderBold(text: string, keyBase: number): React.ReactNode[] {
  return text
    .split(/\*\*(.+?)\*\*/g)
    .map((part, i) => (i % 2 === 1 ? <strong key={`${keyBase}-${i}`}>{part}</strong> : part));
}
