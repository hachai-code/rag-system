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
  distance: number;
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
