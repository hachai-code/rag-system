import { useState } from "react";
import type { CorpusSource, SourcePassage } from "@/lib/types";
import { API_URL } from "@/lib/api";

// The corpus passages the answer cites, as chips at the bottom. Clicking one opens
// that passage in its document — the same /source view the /ask path uses — with the
// retrieved chunk highlighted. The [n] matches the marker in the answer text.
export function CorpusSources({ sources }: { sources: CorpusSource[] }) {
  const [openId, setOpenId] = useState<number | null>(null);
  const [passage, setPassage] = useState<SourcePassage | null>(null);

  async function open(chunkId: number) {
    if (openId === chunkId) {
      setOpenId(null);
      return;
    }
    setOpenId(chunkId);
    setPassage(null);
    const res = await fetch(`${API_URL}/source/${chunkId}`);
    setPassage(await res.json());
  }

  return (
    <section className="mb-8">
      <h2 className="mb-2 text-sm font-medium text-gray-500">Corpus sources</h2>
      <div className="flex flex-wrap gap-2">
        {sources.map((s) => (
          <button
            key={s.chunk_id}
            onClick={() => open(s.chunk_id)}
            className={`rounded-full border px-3 py-1 text-sm hover:bg-gray-100 ${
              openId === s.chunk_id ? "border-black bg-gray-100" : "border-gray-300"
            }`}
          >
            [{s.n}] {s.title}
          </button>
        ))}
      </div>

      {openId !== null && (
        <div className="mt-4 rounded border border-gray-300 p-4">
          {passage === null ? (
            <p className="text-sm text-gray-400">Loading source…</p>
          ) : (
            <>
              <div className="mb-3">
                <div className="font-medium">{passage.title}</div>
                <div className="text-xs text-gray-500">
                  {passage.section} · passage {passage.chunk_index + 1} of {passage.n_chunks}
                </div>
              </div>
              <div className="max-h-96 overflow-y-auto whitespace-pre-wrap text-sm leading-relaxed">
                <span className="text-gray-400">{passage.before}</span>
                {passage.before && "\n"}
                <mark className="bg-yellow-100">{passage.chunk}</mark>
                {passage.after && "\n"}
                <span className="text-gray-400">{passage.after}</span>
              </div>
            </>
          )}
        </div>
      )}
    </section>
  );
}
