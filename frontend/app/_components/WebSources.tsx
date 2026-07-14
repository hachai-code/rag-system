// The deep agent cites web sources as inline URLs in its answer. Pull them out into
// clickable chips linking to each source (corpus points cite [n] inline).
export function WebSources({ answer }: { answer: string }) {
  const urls = [
    ...new Set((answer.match(/https?:\/\/[^\s)\]<>"'`|]+/g) ?? []).map((u) => u.replace(/[.,;]+$/, ""))),
  ].filter((u) => URL.canParse(u));
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
