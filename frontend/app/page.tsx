"use client";

import { useEffect, useRef, useState } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type {
  Citation,
  CorpusSource,
  DeepAgentEvent,
  QAMemory,
  QAMemoryDetail,
  SourcePassage,
  StreamEvent,
} from "@/lib/types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export default function Home() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [citations, setCitations] = useState<Citation[]>([]);
  const [openChip, setOpenChip] = useState<number | null>(null);
  const [source, setSource] = useState<SourcePassage | null>(null);
  const [loading, setLoading] = useState(false);
  const [format, setFormat] = useState<"prose" | "claims">("prose");
  const [model, setModel] = useState<"pro" | "flash">("pro");
  const [topK, setTopK] = useState(25);
  const [deepAgent, setDeepAgent] = useState(false);
  // Max web calls the deep agent may make per question; 0 = unlimited (cap off).
  const [researchBudget, setResearchBudget] = useState(40);
  const [steps, setSteps] = useState<Step[]>([]);
  const [corpusSources, setCorpusSources] = useState<CorpusSource[]>([]);
  const [submitted, setSubmitted] = useState("");
  // One research thread per browser session, so follow-ups reuse the durable
  // notebook the deep agent builds up (multi-turn continuity).
  const [threadId] = useState(() => crypto.randomUUID());
  // The deep agent's stored long-term memories, listed in the sidebar.
  const [memories, setMemories] = useState<QAMemory[]>([]);
  const [memoryKey, setMemoryKey] = useState<string | null>(null);
  const [memory, setMemory] = useState<QAMemoryDetail | null>(null);

  async function loadMemories() {
    const res = await fetch(`${API_URL}/qa`);
    setMemories(await res.json());
  }
  useEffect(() => {
    loadMemories();
  }, []);

  // Open a memory from the sidebar in the main column; clicking it again (or
  // "Back to chat") returns to the chat view, which keeps its state meanwhile.
  async function selectMemory(key: string) {
    if (memoryKey === key) {
      setMemoryKey(null);
      setMemory(null);
      return;
    }
    setMemoryKey(key);
    setMemory(null);
    const res = await fetch(`${API_URL}/qa/${key}`);
    setMemory(await res.json());
  }

  // Keep the newest agent step in view without growing the page: the trace is a
  // fixed-height scroller that follows the tail as steps stream in.
  const traceRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = traceRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [steps]);

  function ask(e: React.FormEvent) {
    e.preventDefault();
    run(format);
  }

  // Pick the answer format for the next question. Each format is a separate backend
  // generation, so this only sets the mode — it takes effect on the next Ask.
  function toggleFormat() {
    setFormat((f) => (f === "prose" ? "claims" : "prose"));
  }

  async function run(fmt: "prose" | "claims") {
    if (!question.trim() || loading) return;
    setAnswer("");
    setCitations([]);
    setOpenChip(null);
    setSource(null);
    setSteps([]);
    setCorpusSources([]);
    setSubmitted(question.trim());
    setLoading(true);

    // The deep agent answers from the corpus then enriches each point with web
    // research. It streams its process — `status` steps as it retrieves, plans,
    // and delegates to the research subagent — then one terminal `answer`.
    if (deepAgent) {
      const res = await fetch(`${API_URL}/ask/agent/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, thread_id: threadId, research_budget: researchBudget }),
      });
      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let done = false;
      while (!done) {
        const chunk = await reader.read();
        done = chunk.done;
        if (!chunk.value) continue;
        buffer += decoder.decode(chunk.value, { stream: true });
        const frames = buffer.split("\n\n");
        buffer = frames.pop() ?? "";
        for (const frame of frames) {
          const data = frame.replace(/^data: /, "");
          if (!data) continue;
          const event: DeepAgentEvent = JSON.parse(data);
          if (event.type === "status") {
            setSteps((prev) => [
              ...prev,
              { callId: event.call_id, scope: event.scope, tool: event.tool, label: event.label },
            ]);
          } else if (event.type === "result") {
            // Attach the tool result to the step it belongs to (matched by call_id).
            setSteps((prev) =>
              prev.map((s) => (s.callId === event.call_id ? { ...s, result: event.preview } : s)),
            );
          } else if (event.type === "sources") {
            setCorpusSources(event.sources);
          } else if (event.type === "answer") {
            setAnswer(event.text);
          } else if (event.type === "error") {
            setAnswer(`Error: ${event.message}`);
          }
        }
      }
      setLoading(false);
      loadMemories(); // the finished run may have stored a new memory
      return;
    }

    const res = await fetch(`${API_URL}/ask/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, format: fmt, model, top_k: topK }),
    });

    // Read the SSE stream: frames are separated by a blank line, each one a
    // "data: {json}" line. We buffer bytes and parse whole frames as they arrive.
    const reader = res.body!.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let done = false;
    while (!done) {
      const chunk = await reader.read();
      done = chunk.done;
      if (!chunk.value) continue;
      buffer += decoder.decode(chunk.value, { stream: true });
      const frames = buffer.split("\n\n");
      buffer = frames.pop() ?? ""; // keep the trailing partial frame for next read
      for (const frame of frames) {
        const data = frame.replace(/^data: /, "");
        if (!data) continue;
        const event: StreamEvent = JSON.parse(data);
        if (event.type === "text") {
          setAnswer((prev) => prev + event.text);
        } else {
          setCitations((prev) => [...prev, event]);
        }
      }
    }
    setLoading(false);
  }

  // Open a citation: fetch the chunk's place in its document and show it. Clicking
  // the open chip again closes the panel.
  async function openSource(i: number, chunkId: number) {
    if (openChip === i) {
      setOpenChip(null);
      return;
    }
    setOpenChip(i);
    setSource(null);
    const res = await fetch(`${API_URL}/source/${chunkId}`);
    setSource(await res.json());
  }

  return (
    <div className="flex">
      <MemorySidebar memories={memories} selectedKey={memoryKey} onSelect={selectMemory} />
      <main className="mx-auto max-w-4xl flex-1 px-4 py-10">

      {memoryKey !== null ? (
        memory === null ? (
          <p className="text-sm text-gray-400">Loading memory…</p>
        ) : (
          <MemoryViewer memory={memory} onBack={() => selectMemory(memoryKey)} />
        )
      ) : (
        <>

      <div className="mb-2 flex items-center justify-end gap-4 text-sm text-gray-400">
        {!deepAgent && (
          <>
        <label
          title="generation model"
          className="flex items-center gap-1.5 rounded-full bg-gray-100 py-1 pl-3 pr-1.5 transition-colors focus-within:bg-gray-200 hover:bg-gray-200"
        >
          <span className="font-mono text-xs uppercase tracking-wide text-gray-500">model</span>
          <select
            value={model}
            onChange={(e) => setModel(e.target.value as "pro" | "flash")}
            className="rounded-full bg-white py-0.5 pl-2 pr-1 font-medium text-gray-700 shadow-sm focus:outline-none focus:ring-1 focus:ring-gray-300"
          >
            <option value="pro">pro</option>
            <option value="flash">flash</option>
          </select>
        </label>
        <label
          title="top_k — chunks retrieved and handed to the generator"
          className="flex items-center gap-1.5 rounded-full bg-gray-100 py-1 pl-3 pr-1.5 transition-colors focus-within:bg-gray-200 hover:bg-gray-200"
        >
          <span className="font-mono text-xs uppercase tracking-wide text-gray-500">top_k</span>
          <input
            type="number"
            min={1}
            value={topK}
            onChange={(e) => setTopK(Number(e.target.value))}
            className="w-12 rounded-full bg-white py-0.5 text-center font-medium text-gray-700 shadow-sm focus:outline-none focus:ring-1 focus:ring-gray-300"
          />
        </label>
        <button
          onClick={toggleFormat}
          disabled={loading}
          className="text-blue-600 hover:underline disabled:opacity-50"
        >
          {format === "prose" ? "Switch to claims + citations" : "Switch to prose"}
        </button>
          </>
        )}
        {deepAgent && (
          <label
            title="web calls the deep agent may make per question (0 = unlimited)"
            className="flex items-center gap-1.5 rounded-full bg-gray-100 py-1 pl-3 pr-1.5 transition-colors focus-within:bg-gray-200 hover:bg-gray-200"
          >
            <span className="font-mono text-xs uppercase tracking-wide text-gray-500">web calls</span>
            <input
              type="number"
              min={0}
              value={researchBudget}
              onChange={(e) => setResearchBudget(Number(e.target.value))}
              className="w-12 rounded-full bg-white py-0.5 text-center font-medium text-gray-700 shadow-sm focus:outline-none focus:ring-1 focus:ring-gray-300"
            />
          </label>
        )}
        <button
          onClick={() => setDeepAgent((v) => !v)}
          disabled={loading}
          title="Answer from the corpus, then enrich each point with external web research (slower)"
          className={`rounded-full px-3 py-1 text-sm font-medium transition-colors disabled:opacity-50 ${
            deepAgent
              ? "bg-purple-600 text-white hover:bg-purple-700"
              : "bg-gray-100 text-gray-600 hover:bg-gray-200"
          }`}
        >
          Deep Agent {deepAgent ? "on" : "off"}
        </button>
      </div>

      <form onSubmit={ask} className="mb-8 flex gap-2">
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask about the corpus…"
          className="flex-1 rounded border border-gray-300 px-3 py-2 focus:outline-none focus:ring"
        />
        <button
          type="submit"
          disabled={loading}
          className="rounded bg-black px-4 py-2 text-white disabled:opacity-50"
        >
          {loading ? "…" : "Ask"}
        </button>
      </form>

      {submitted && (
        <div className="mb-6 flex justify-end">
          <div className="max-w-[80%] whitespace-pre-wrap rounded-2xl rounded-br-sm bg-gray-100 px-4 py-2 text-gray-800">
            {submitted}
          </div>
        </div>
      )}

      {steps.length > 0 && (
        <section className="mb-6">
          <h2 className="mb-2 text-sm font-medium text-gray-500">Reasoning</h2>
          <div
            ref={traceRef}
            className="max-h-72 space-y-1.5 overflow-y-auto border-l-2 border-gray-200 pl-4"
          >
            {steps.map((s, i) => (
              <TraceStep key={s.callId || i} step={s} active={loading && i === steps.length - 1} />
            ))}
          </div>
        </section>
      )}

      {answer &&
        (deepAgent ? (
          // The deep agent answers in Markdown (headings, quotes, tables, links);
          // render it as such. The /ask path stays on AnswerBody for its inline,
          // span-anchored citation markers, which Markdown can't express.
          <article className="prose prose-neutral mb-8 max-w-none">
            <Markdown remarkPlugins={[remarkGfm]}>{answer}</Markdown>
          </article>
        ) : (
          <article className="mb-8 whitespace-pre-wrap leading-relaxed">
            <AnswerBody answer={answer} citations={citations} openSource={openSource} />
          </article>
        ))}

      {deepAgent && corpusSources.length > 0 && <CorpusSources sources={corpusSources} />}

      {deepAgent && answer && <WebSources answer={answer} />}

      {citations.length > 0 && (
        <section>
          <h2 className="mb-2 text-sm font-medium text-gray-500">Citations</h2>
          <div className="flex flex-wrap gap-2">
            {citations.map((c, i) => (
              <button
                key={i}
                onClick={() => openSource(i, c.chunk_id)}
                className={`rounded-full border px-3 py-1 text-sm hover:bg-gray-100 ${
                  openChip === i ? "border-black bg-gray-100" : "border-gray-300"
                }`}
              >
                [{i + 1}] {c.title}
              </button>
            ))}
          </div>

          {openChip !== null && (
            <div className="mt-4 rounded border border-gray-300 p-4">
              {source === null ? (
                <p className="text-sm text-gray-400">Loading source…</p>
              ) : (
                <>
                  <div className="mb-3">
                    <div className="font-medium">{source.title}</div>
                    <div className="text-xs text-gray-500">
                      {source.section} · passage {source.chunk_index + 1} of{" "}
                      {source.n_chunks}
                    </div>
                  </div>
                  <div className="max-h-96 overflow-y-auto whitespace-pre-wrap text-sm leading-relaxed">
                    <span className="text-gray-400">{source.before}</span>
                    {source.before && "\n"}
                    <Highlight
                      chunk={source.chunk}
                      cited={citations[openChip].cited_text}
                    />
                    {source.after && "\n"}
                    <span className="text-gray-400">{source.after}</span>
                  </div>
                </>
              )}
            </div>
          )}
        </section>
      )}
        </>
      )}
    </main>
    </div>
  );
}

// A stored memory opened from the sidebar: the full cached record — question,
// markdown answer, corpus sources, web sources, and the research subagent's notes.
function MemoryViewer({ memory, onBack }: { memory: QAMemoryDetail; onBack: () => void }) {
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

// The deep agent's long-term memories (its Q&A cache), newest first. Selecting one
// will open the full record in the main column. Hidden on small screens.
// ponytail: no mobile drawer — add one if the public UI needs it.
function MemorySidebar({
  memories,
  selectedKey,
  onSelect,
}: {
  memories: QAMemory[];
  selectedKey: string | null;
  onSelect: (key: string) => void;
}) {
  return (
    <aside className="hidden w-72 shrink-0 border-r border-gray-200 md:block">
      <div className="sticky top-0 h-screen overflow-y-auto p-4">
        <h2 className="mb-3 text-sm font-medium text-gray-500">Memory</h2>
        {memories.length === 0 && (
          <p className="text-sm text-gray-400">No memories yet — answered questions land here.</p>
        )}
        <div className="space-y-1">
          {memories.map((m) => (
            <button
              key={m.key}
              onClick={() => onSelect(m.key)}
              className={`block w-full rounded border px-3 py-2 text-left text-sm hover:bg-gray-100 ${
                selectedKey === m.key ? "border-black bg-gray-100" : "border-transparent"
              }`}
            >
              <span className="line-clamp-2">{m.question}</span>
              <span className="mt-0.5 block text-xs text-gray-400">
                {new Date(m.created_at).toLocaleDateString()}
              </span>
            </button>
          ))}
        </div>
      </div>
    </aside>
  );
}

// One tool call in the agent's live trace, plus its result once it arrives.
type Step = { callId: string; scope: string; tool: string; label: string; result?: string };

// A step in the reasoning trace: a tool badge + summary, collapsible to reveal a
// preview of what the tool returned. Research-subagent steps are indented. Until
// the result lands the step is a plain line (nothing to expand); the active step
// shows a ▸ marker while the agent works.
function TraceStep({ step, active }: { step: Step; active: boolean }) {
  const indent = step.scope === "research" ? "ml-4" : "";
  const head = (
    <>
      <span className="rounded bg-gray-100 px-1.5 py-0.5 font-mono text-[11px] text-gray-500">
        {step.tool}
      </span>{" "}
      <span className="text-gray-600">{step.label}</span>
      {active && !step.result && <span className="text-gray-300"> ▸</span>}
    </>
  );
  if (!step.result) {
    return <div className={`text-sm ${indent}`}>{head}</div>;
  }
  return (
    <details className={`text-sm ${indent}`}>
      <summary className="cursor-pointer marker:text-gray-300">{head}</summary>
      <pre className="ml-4 mt-1 max-h-48 overflow-y-auto whitespace-pre-wrap rounded bg-gray-50 p-2 text-xs text-gray-500">
        {step.result}
      </pre>
    </details>
  );
}

// The corpus passages the answer cites, as chips at the bottom. Clicking one opens
// that passage in its document — the same /source view the /ask path uses — with the
// retrieved chunk highlighted. The [n] matches the marker in the answer text.
function CorpusSources({ sources }: { sources: CorpusSource[] }) {
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

// The deep agent cites web sources as inline URLs in its answer. Pull them out into
// clickable chips linking to each source (corpus points cite [n] inline).
function WebSources({ answer }: { answer: string }) {
  const urls = [
    ...new Set((answer.match(/https?:\/\/[^\s)\]<>"']+/g) ?? []).map((u) => u.replace(/[.,;]+$/, ""))),
  ];
  if (urls.length === 0) return null;
  return (
    <section className="mb-8">
      <h2 className="mb-2 text-sm font-medium text-gray-500">Web sources</h2>
      <div className="flex flex-wrap gap-2">
        {urls.map((u, i) => (
          <a
            key={u}
            href={u}
            target="_blank"
            rel="noopener noreferrer"
            className="rounded-full border border-gray-300 px-3 py-1 text-sm text-blue-600 hover:bg-gray-100"
          >
            [{i + 1}] {new URL(u).hostname.replace(/^www\./, "")}
          </a>
        ))}
      </div>
    </section>
  );
}

// Render the answer with **bold** and inline citation markers placed right after
// the claim each citation backs. Each claim is an exact, in-order span of the
// answer (Anthropic citations), so we locate them sequentially; citations sharing
// a span cluster into one [n][m] group. The marker number matches the chip below.
function AnswerBody({
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

// The retrieved chunk, highlighted within its document; the exact cited sentence
// is highlighted more strongly inside it.
function Highlight({ chunk, cited }: { chunk: string; cited: string }) {
  const at = chunk.indexOf(cited);
  if (at === -1) return <mark className="bg-yellow-100">{chunk}</mark>;
  return (
    <mark className="bg-yellow-100">
      {chunk.slice(0, at)}
      <mark className="bg-yellow-300 font-medium">{cited}</mark>
      {chunk.slice(at + cited.length)}
    </mark>
  );
}
