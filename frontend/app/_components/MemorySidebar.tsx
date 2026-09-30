import { useState } from "react";
import type { QAMemory } from "@/lib/types";

// The deep agent's long-term memories (its Q&A cache), newest first. Selecting one
// will open the full record in the main column. On small screens the list lives
// behind a floating "Memory" button that opens it as a full-screen overlay.
export function MemorySidebar({
  memories,
  selectedKey,
  onSelect,
}: {
  memories: QAMemory[] | null; // null = still fetching
  selectedKey: string | null;
  onSelect: (key: string) => void;
}) {
  const [open, setOpen] = useState(false);

  function select(key: string) {
    setOpen(false); // close the mobile overlay so the opened memory is visible
    onSelect(key);
  }

  const list = (
    <>
      {memories === null && <p className="animate-pulse text-sm text-gray-400">Loading…</p>}
      {memories?.length === 0 && (
        <p className="text-sm text-gray-400">No memories yet — answered questions land here.</p>
      )}
      <div className="space-y-1">
        {(memories ?? []).map((m) => (
          <button
            key={m.key}
            onClick={() => select(m.key)}
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
    </>
  );

  return (
    <>
      <aside className="hidden w-72 shrink-0 border-r border-gray-200 md:block">
        <div className="sticky top-0 h-screen overflow-y-auto p-4">
          <h2 className="mb-3 text-sm font-medium text-gray-500">Memory</h2>
          {list}
        </div>
      </aside>

      <button
        onClick={() => setOpen(true)}
        className="fixed bottom-4 right-4 z-10 rounded-full bg-black px-4 py-2 text-sm font-medium text-white shadow-lg md:hidden"
      >
        Memory
      </button>

      {open && (
        <div className="fixed inset-0 z-20 overflow-y-auto bg-white p-4 md:hidden">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-sm font-medium text-gray-500">Memory</h2>
            <button onClick={() => setOpen(false)} className="text-sm text-blue-600">
              Close
            </button>
          </div>
          {list}
        </div>
      )}
    </>
  );
}
