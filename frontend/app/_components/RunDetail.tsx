"use client";

import { useEffect, useState } from "react";
// Aliased: the generated name collides with this component.
import type { RunDetail as EvalRunDetail } from "@/lib/types";
import { API_URL } from "@/lib/api";

// Drill-down for one eval run: every judged question with its per-dimension
// PASS/FAIL and the judge's rationale. Fetched on demand when a run is opened.
export function RunDetail({ runId, onClose }: { runId: number; onClose: () => void }) {
  const [detail, setDetail] = useState<EvalRunDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Parent keys this component on runId, so a different run remounts with fresh
  // state — no need to reset detail/error here (which would be a synchronous
  // setState in an effect).
  useEffect(() => {
    fetch(`${API_URL}/evals/run/${runId}`)
      .then((res) => (res.ok ? res.json() : Promise.reject(new Error(`${res.status}`))))
      .then(setDetail)
      .catch((e) => setError(String(e)));
  }, [runId]);

  return (
    <section className="mt-4 rounded-lg border border-gray-200 p-4 dark:border-gray-700">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-base font-semibold">Run #{runId} — rationales</h3>
        <button onClick={onClose} className="text-sm text-blue-600 hover:underline">
          Close
        </button>
      </div>

      {error && <p className="text-sm text-gray-500">Failed to load run #{runId}: {error}</p>}
      {!detail && !error && <p className="text-sm text-gray-500">Loading…</p>}

      {detail && (
        <div className="space-y-2">
          {detail.results.map((row) => (
            <details key={row.question_id} className="rounded border border-gray-100 dark:border-gray-800">
              <summary className="flex cursor-pointer flex-wrap items-center gap-2 px-3 py-2 text-sm">
                <span className="font-mono text-xs text-gray-400">#{row.question_id}</span>
                {row.split && (
                  <span className="rounded bg-gray-100 px-1.5 py-0.5 text-xs text-gray-500 dark:bg-gray-800">
                    {row.split}
                  </span>
                )}
                {Object.entries(row.scores).map(([dim, passed]) => (
                  <span
                    key={dim}
                    title={passed ? "PASS" : "FAIL"}
                    className={`rounded px-1.5 py-0.5 font-mono text-xs ${
                      passed
                        ? "bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-300"
                        : "bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-300"
                    }`}
                  >
                    {dim}
                  </span>
                ))}
                <span className="truncate text-gray-600 dark:text-gray-300">{row.question}</span>
              </summary>

              <div className="space-y-3 px-3 pb-3 pt-1 text-sm">
                {Object.entries(row.rationales).map(([dim, rationale]) => (
                  <div key={dim}>
                    <span className={`font-mono text-xs ${row.scores[dim] ? "text-green-600" : "text-red-600"}`}>
                      {dim} {row.scores[dim] ? "PASS" : "FAIL"}
                    </span>
                    <p className="text-gray-600 dark:text-gray-300">{rationale}</p>
                  </div>
                ))}
                <details className="text-gray-500">
                  <summary className="cursor-pointer text-xs">Answer judged</summary>
                  <p className="mt-1 whitespace-pre-wrap text-gray-500">{row.answer}</p>
                </details>
              </div>
            </details>
          ))}
        </div>
      )}
    </section>
  );
}
