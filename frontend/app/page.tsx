"use client";

import { useState } from "react";
import type { Citation, DeepAgentEvent, SourcePassage, StreamEvent } from "@/lib/types";

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
  const [steps, setSteps] = useState<{ scope: string; text: string }[]>([]);
  // One research thread per browser session, so follow-ups reuse the durable
  // notebook the deep agent builds up (multi-turn continuity).
  const [threadId] = useState(() => crypto.randomUUID());

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
    setLoading(true);

    // The deep agent answers from the corpus then enriches each point with web
    // research. It streams its process — `status` steps as it retrieves, plans,
    // and delegates to the research subagent — then one terminal `answer`.
    if (deepAgent) {
      const res = await fetch(`${API_URL}/ask/agent/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, thread_id: threadId }),
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
            setSteps((prev) => [...prev, { scope: event.scope, text: event.text }]);
          } else if (event.type === "answer") {
            setAnswer(event.text);
          } else {
            setAnswer(`Error: ${event.message}`);
          }
        }
      }
      setLoading(false);
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
    <main className="mx-auto max-w-4xl px-4 py-10">

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

      {steps.length > 0 && (
        <ol className="mb-6 space-y-1 border-l-2 border-gray-200 pl-4 text-sm text-gray-500">
          {steps.map((s, i) => (
            <li key={i} className={s.scope === "research" ? "pl-4 text-gray-400" : ""}>
              {loading && i === steps.length - 1 ? "▸ " : "· "}
              {s.text}
            </li>
          ))}
        </ol>
      )}

      {answer && (
        <article className="mb-8 whitespace-pre-wrap leading-relaxed">
          <AnswerBody answer={answer} citations={citations} openSource={openSource} />
        </article>
      )}

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
    </main>
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
