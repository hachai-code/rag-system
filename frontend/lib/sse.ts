// Reading an SSE stream: frames are separated by a blank line, each carrying one
// "data: {json}" line. We buffer bytes and parse whole frames as they arrive.
//
// fetch + reader rather than the browser's EventSource: two of the three streams are
// POSTs with a body, and the deep-agent stream reconnects with its own cursor.
export async function* sseEvents<T>(res: Response): AsyncGenerator<T> {
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
      // A frame with no data line is a keep-alive comment (": ping - …", which
      // sse-starlette sends every 15s) — skip it rather than parse it as an event.
      const data = frame
        .split("\n")
        .filter((line) => line.startsWith("data: "))
        .map((line) => line.slice("data: ".length))
        .join("\n");
      if (data) yield JSON.parse(data) as T;
    }
  }
}
