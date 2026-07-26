import { useState } from "react";
import type { CorpusSource } from "@/lib/types";
import { SourcePanel } from "./SourcePanel";

// The corpus passages the answer cites, as chips at the bottom. Clicking one opens
// that passage in its document — the same /source view the /ask path uses — with the
// retrieved chunk highlighted. The [n] matches the marker in the answer text.
export function CorpusSources({ sources }: { sources: CorpusSource[] }) {
  const [openId, setOpenId] = useState<number | null>(null);

  return (
    <section className="mb-8">
      <h2 className="mb-2 text-sm font-medium text-gray-500">Corpus sources</h2>
      <div className="flex flex-wrap gap-2">
        {sources.map((s) => (
          <button
            key={s.chunk_id}
            onClick={() => setOpenId((open) => (open === s.chunk_id ? null : s.chunk_id))}
            className={`rounded-full border px-3 py-1 text-sm hover:bg-gray-100 ${
              openId === s.chunk_id ? "border-black bg-gray-100" : "border-gray-300"
            }`}
          >
            [{s.n}] {s.title}
          </button>
        ))}
      </div>

      {openId !== null && <SourcePanel chunkId={openId} />}
    </section>
  );
}
