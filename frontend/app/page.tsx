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
import { API_URL } from "@/lib/api";
import { sseEvents } from "@/lib/sse";
import { AnswerBody } from "./_components/AnswerBody";
import { CorpusSources } from "./_components/CorpusSources";
import { Highlight } from "./_components/Highlight";
import { MemorySidebar } from "./_components/MemorySidebar";
import { MemoryViewer } from "./_components/MemoryViewer";
import { TraceStep, type Step } from "./_components/TraceStep";
import { WebSources } from "./_components/WebSources";

// Resolves once the tab is visible (mobile browsers kill connections of
// backgrounded tabs), plus a beat for the network to come back.
function visibleAgain(): Promise<void> {
  return new Promise((resolve) => {
    const settle = () => setTimeout(resolve, 1000);
    if (document.visibilityState === "visible") {
      settle();
      return;
    }
    const onVisible = () => {
      if (document.visibilityState !== "visible") return;
      document.removeEventListener("visibilitychange", onVisible);
      settle();
    };
    document.addEventListener("visibilitychange", onVisible);
  });
}

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
  // null = still fetching (sidebar shows a loader instead of the empty state).
  const [memories, setMemories] = useState<QAMemory[] | null>(null);
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
    // research. The run executes server-side, decoupled from this connection: we
    // start it, then read its buffered event stream with a cursor. If the phone
    // backgrounds the tab and the connection dies, we reconnect from where we left
    // off once the tab is visible again — the run kept going the whole time.
    if (deepAgent) {
      const start = await fetch(`${API_URL}/ask/agent/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, thread_id: threadId, research_budget: researchBudget }),
      });
      const { run_id } = await start.json();

      let received = 0;
      for (;;) {
        try {
          const res = await fetch(`${API_URL}/ask/agent/run/${run_id}?after=${received}`);
          if (res.status === 404) {
            setAnswer("Error: this run is gone (server restarted). Ask again.");
            break;
          }
          for await (const event of sseEvents<DeepAgentEvent>(res)) {
            received += 1;
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
          break; // the server only closes the stream when the run is done
        } catch {
          await visibleAgain(); // connection died (likely backgrounded) — reconnect
        }
      }
      setLoading(false);
      loadMemories(); // the finished run may have stored a new memory
      return;
    }

    // The /ask path streams the generation live, so a dropped connection loses the
    // rest of the answer — surface that instead of hanging on "…" forever.
    // ponytail: no resume here; the deep-agent path has the durable-run treatment.
    try {
      const res = await fetch(`${API_URL}/ask/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, format: fmt, model, top_k: topK }),
      });

      for await (const event of sseEvents<StreamEvent>(res)) {
        if (event.type === "text") {
          setAnswer((prev) => prev + event.text);
        } else {
          setCitations((prev) => [...prev, event]);
        }
      }
    } catch {
      setAnswer((prev) => prev + "\n\n[Connection lost — ask again to retry.]");
    } finally {
      setLoading(false);
    }
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
