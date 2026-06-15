"use client";

import { useState } from "react";
import type { Citation, StreamEvent } from "@/lib/types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export default function Home() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [citations, setCitations] = useState<Citation[]>([]);
  const [openChip, setOpenChip] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);

  async function ask(e: React.FormEvent) {
    e.preventDefault();
    if (!question.trim() || loading) return;
    setAnswer("");
    setCitations([]);
    setOpenChip(null);
    setLoading(true);

    const res = await fetch(`${API_URL}/ask/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
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

  return (
    <main className="mx-auto max-w-2xl px-4 py-10">
      <h1 className="mb-6 text-2xl font-semibold">innerdance RAG</h1>

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

      {answer && (
        <article className="mb-8 whitespace-pre-wrap leading-relaxed">{answer}</article>
      )}

      {citations.length > 0 && (
        <section>
          <h2 className="mb-2 text-sm font-medium text-gray-500">Citations</h2>
          <div className="flex flex-wrap gap-2">
            {citations.map((c, i) => (
              <button
                key={i}
                onClick={() => setOpenChip(openChip === i ? null : i)}
                className="rounded-full border border-gray-300 px-3 py-1 text-sm hover:bg-gray-100"
              >
                [{i + 1}] {c.title}
              </button>
            ))}
          </div>
          {openChip !== null && (
            <blockquote className="mt-3 border-l-2 border-gray-300 pl-3 text-sm text-gray-600">
              “{citations[openChip].cited_text}”
              <div className="mt-1 text-xs text-gray-400">
                {citations[openChip].source}
              </div>
            </blockquote>
          )}
        </section>
      )}
    </main>
  );
}
