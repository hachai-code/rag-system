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
