import type { QAMemory } from "@/lib/types";

// The deep agent's long-term memories (its Q&A cache), newest first. Selecting one
// will open the full record in the main column. Hidden on small screens.
// ponytail: no mobile drawer — add one if the public UI needs it.
export function MemorySidebar({
  memories,
  selectedKey,
  onSelect,
}: {
  memories: QAMemory[] | null; // null = still fetching
  selectedKey: string | null;
  onSelect: (key: string) => void;
}) {
  return (
    <aside className="hidden w-72 shrink-0 border-r border-gray-200 md:block">
      <div className="sticky top-0 h-screen overflow-y-auto p-4">
        <h2 className="mb-3 text-sm font-medium text-gray-500">Memory</h2>
        {memories === null && <p className="animate-pulse text-sm text-gray-400">Loading…</p>}
        {memories?.length === 0 && (
          <p className="text-sm text-gray-400">No memories yet — answered questions land here.</p>
        )}
        <div className="space-y-1">
          {(memories ?? []).map((m) => (
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
