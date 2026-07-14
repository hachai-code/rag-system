// Mirrors the Pydantic models in app.py. Keep these in sync with that file.

export interface Citation {
  claim: string; // the span of the answer this citation backs
  cited_text: string; // the exact source quote, extracted by the API
  chunk_id: number;
  title: string;
  source: string;
}

export interface Source {
  title: string;
  source: string;
  distance: number | null; // null for keyword-only hits (no vector distance)
}

// Returned by POST /ask (non-streaming).
export interface AskResponse {
  answer: string;
  citations: Citation[];
  sources: Source[];
}

// Server-Sent Events from POST /ask/stream: answer text arrives as `text`
// events, then one `citation` event per source once the message completes.
export type StreamEvent =
  | { type: "text"; text: string }
  | ({ type: "citation" } & Citation);

// Returned by POST /ask/agent: the deep agent's answer, citations inline in the
// text. thread_id keys the durable research thread for multi-turn follow-ups.
export interface DeepAgentResponse {
  answer: string;
  thread_id: string;
}

// A corpus passage the deep agent's answer cites, listed as a chip at the bottom.
// `n` is its stable [n] in the answer; `chunk_id` opens it via GET /source/{chunk_id}.
export interface CorpusSource {
  n: number;
  chunk_id: number;
  title: string;
  source: string;
}

// SSE from GET /ask/agent/run/{run_id}: a `status` event per tool call and a `result`
// event per tool result (correlated by call_id) stream in as the agent works
// (scope "research" = inside the web-research subagent), then a `sources` event
// listing the cited corpus passages and one terminal `answer` — or `error`.
export type DeepAgentEvent =
  | { type: "status"; scope: "main" | "research"; call_id: string; tool: string; label: string }
  | { type: "result"; call_id: string; preview: string }
  | { type: "sources"; sources: CorpusSource[] }
  | { type: "answer"; text: string; thread_id: string }
  | { type: "error"; message: string };

// Returned by GET /qa: one of the deep agent's stored long-term memories
// (Q&A cache records), list view — newest first.
export interface QAMemory {
  key: string;
  question: string;
  created_at: string; // ISO timestamp
}

// Returned by GET /qa/{key}: the full stored record.
export interface QAMemoryDetail extends QAMemory {
  answer: string;
  corpus_sources: CorpusSource[];
  web_urls: string[];
  research_files: Record<string, string>; // path -> research note text
}

// Returned by GET /source/{chunk_id}: the cited chunk in its document context.
export interface SourcePassage {
  title: string;
  section: string;
  chunk_index: number;
  n_chunks: number;
  before: string; // context preceding the cited chunk
  chunk: string; // the retrieved chunk, to highlight
  after: string; // context following the cited chunk
}
