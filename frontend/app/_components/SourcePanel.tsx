import { useEffect, useState } from "react";
import type { SourcePassage } from "@/lib/types";
import { API_URL } from "@/lib/api";
import { Highlight } from "./Highlight";

// A cited chunk shown in its place in the document: dimmed context either side, the
// chunk highlighted — and, when the citation carries the exact quote, that quote
// highlighted more strongly inside it. Owns its own fetch; parents only pick the chunk.
export function SourcePanel({ chunkId, cited }: { chunkId: number; cited?: string }) {
  const [source, setSource] = useState<SourcePassage | null>(null);

  useEffect(() => {
    let stale = false;
    setSource(null);
    fetch(`${API_URL}/source/${chunkId}`)
      .then((res) => res.json())
      .then((s) => {
        if (!stale) setSource(s);
      });
    return () => {
      stale = true;
    };
  }, [chunkId]);

  return (
    <div className="mt-4 rounded border border-gray-300 p-4">
      {source === null ? (
        <p className="text-sm text-gray-400">Loading source…</p>
      ) : (
        <>
          <div className="mb-3">
            <div className="font-medium">{source.title}</div>
            <div className="text-xs text-gray-500">
              {source.section} · passage {source.chunk_index + 1} of {source.n_chunks}
            </div>
          </div>
          <div className="max-h-96 overflow-y-auto whitespace-pre-wrap text-sm leading-relaxed">
            <span className="text-gray-400">{source.before}</span>
            {source.before && "\n"}
            {cited ? (
              <Highlight chunk={source.chunk} cited={cited} />
            ) : (
              <mark className="bg-yellow-100">{source.chunk}</mark>
            )}
            {source.after && "\n"}
            <span className="text-gray-400">{source.after}</span>
          </div>
        </>
      )}
    </div>
  );
}
