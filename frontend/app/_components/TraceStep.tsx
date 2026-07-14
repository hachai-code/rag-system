// One tool call in the agent's live trace, plus its result once it arrives.
export type Step = { callId: string; scope: string; tool: string; label: string; result?: string };

// A step in the reasoning trace: a tool badge + summary, collapsible to reveal a
// preview of what the tool returned. Research-subagent steps are indented. Until
// the result lands the step is a plain line (nothing to expand); the active step
// shows a ▸ marker while the agent works.
export function TraceStep({ step, active }: { step: Step; active: boolean }) {
  const indent = step.scope === "research" ? "ml-4" : "";
  const head = (
    <>
      <span className="rounded bg-gray-100 px-1.5 py-0.5 font-mono text-[11px] text-gray-500">
        {step.tool}
      </span>{" "}
      <span className="text-gray-600">{step.label}</span>
      {active && !step.result && <span className="text-gray-300"> ▸</span>}
    </>
  );
  if (!step.result) {
    return <div className={`text-sm ${indent}`}>{head}</div>;
  }
  return (
    <details className={`text-sm ${indent}`}>
      <summary className="cursor-pointer marker:text-gray-300">{head}</summary>
      <pre className="ml-4 mt-1 max-h-48 overflow-y-auto whitespace-pre-wrap rounded bg-gray-50 p-2 text-xs text-gray-500">
        {step.result}
      </pre>
    </details>
  );
}
