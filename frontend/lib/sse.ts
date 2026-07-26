// Parse a fetch() Response's SSE body: frames are separated by a blank line, each one
// a "data: {json}" line. Buffers bytes and yields one parsed event per whole frame.
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
    buffer = frames.pop() ?? ""; // keep the trailing partial frame for the next read
    for (const frame of frames) {
      const data = frame.replace(/^data: /, "");
      if (!data) continue;
      yield JSON.parse(data) as T;
    }
  }
}
