import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { QAMemoryDetail } from "@/lib/types";
import { CorpusSources } from "./CorpusSources";
import { WebSources } from "./WebSources";

// A stored memory opened from the sidebar: the full cached record — question,
// markdown answer, corpus sources, web sources, and the research subagent's notes.
export function MemoryViewer({ memory, onBack }: { memory: QAMemoryDetail; onBack: () => void }) {
  return (
    <div>
      <button onClick={onBack} className="mb-6 text-sm text-blue-600 hover:underline">
        ← Back to chat
      </button>
      <h1 className="mb-1 text-xl font-medium">{memory.question}</h1>
      <div className="mb-6 text-xs text-gray-500">
        Remembered {new Date(memory.created_at).toLocaleString()}
      </div>
      <article className="prose prose-neutral mb-8 max-w-none">
        <Markdown remarkPlugins={[remarkGfm]}>{memory.answer}</Markdown>
      </article>
      {memory.corpus_sources.length > 0 && <CorpusSources sources={memory.corpus_sources} />}
      <WebSources answer={memory.answer} />
      {Object.keys(memory.research_files).length > 0 && (
        <section className="mb-8">
          <h2 className="mb-2 text-sm font-medium text-gray-500">Research notes</h2>
          <div className="space-y-1.5">
            {Object.entries(memory.research_files).map(([path, text]) => (
              <details key={path} className="text-sm">
                <summary className="cursor-pointer font-mono text-xs text-gray-600 marker:text-gray-300">
                  {path}
                </summary>
                <pre className="mt-1 max-h-72 overflow-y-auto whitespace-pre-wrap rounded bg-gray-50 p-2 text-xs text-gray-500">
                  {text}
                </pre>
              </details>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
