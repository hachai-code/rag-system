"use client";

import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { EvalsSummary } from "@/lib/types";
import { API_URL } from "@/lib/api";
import { RunDetail } from "../_components/RunDetail";

// Categorical palette (dataviz reference instance, CVD-validated): one fixed
// slot per rubric dimension so a dimension keeps its hue across all charts and
// runs. Dark steps are the same hues re-stepped for the dark surface.
const VIZ_VARS = `
.viz-root {
  --dim-A: #2a78d6; --dim-B: #1baf7a; --dim-C: #eda100;
  --dim-D: #008300; --dim-E: #4a3aa7; --dim-F: #e34948;
  --seq: #2a78d6;
  --split-dev: #2a78d6; --split-test: #1baf7a;
  --grid: #e1e0d9; --muted: #898781;
}
@media (prefers-color-scheme: dark) {
  .viz-root {
    --dim-A: #3987e5; --dim-B: #199e70; --dim-C: #c98500;
    --dim-D: #008300; --dim-E: #9085e9; --dim-F: #e66767;
    --seq: #3987e5;
    --split-dev: #3987e5; --split-test: #199e70;
    --grid: #2c2c2a;
  }
}
`;

const DIM_COLOR = (dim: string) => `var(--dim-${dim}, var(--muted))`;
const pct = (v: number) => `${Math.round(v * 100)}%`;
const usd = (v: number) => `$${v.toFixed(4)}`;

const TOOLTIP_STYLE = {
  background: "var(--background)",
  color: "var(--foreground)",
  border: "1px solid var(--grid)",
  borderRadius: "6px",
  fontSize: "12px",
};

function Section({ title, sub, children }: { title: string; sub: string; children: React.ReactNode }) {
  return (
    <section className="mb-10">
      <h2 className="text-lg font-semibold">{title}</h2>
      <p className="mb-3 text-sm text-gray-500">{sub}</p>
      {children}
    </section>
  );
}

export default function EvalsPage() {
  const [data, setData] = useState<EvalsSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [openRun, setOpenRun] = useState<number | null>(null);

  useEffect(() => {
    fetch(`${API_URL}/evals/summary`)
      .then((res) => (res.ok ? res.json() : Promise.reject(new Error(`${res.status}`))))
      .then(setData)
      .catch((e) => setError(String(e)));
  }, []);

  if (error) return <main className="mx-auto max-w-4xl px-4 py-10">Failed to load eval data from {API_URL}: {error}</main>;
  if (!data) return <main className="mx-auto max-w-4xl px-4 py-10 text-gray-500">Loading…</main>;
  if (data.runs.length === 0)
    return (
      <main className="mx-auto max-w-4xl px-4 py-10">
        No full eval runs yet (partial <code>--limit</code> runs are hidden) —{" "}
        <code>uv run python -m evals.run --config evals/configs/baseline.json --split dev</code>
      </main>
    );

  const { runs, final } = data;
  // Every dimension seen anywhere, in rubric order — the color slot follows the
  // dimension letter, so a filtered/missing dimension never repaints the others.
  const dims = [...new Set(runs.flatMap((r) => Object.keys(r.pass_rate)))].sort();

  const trend = runs.map((r) => ({
    label: `#${r.run_id}`,
    date: r.created_at.slice(0, 10),
    config: r.config_name ?? `run ${r.run_id}`,
    cost: r.cost,
    n: r.n,
    ...r.pass_rate,
  }));

  const splitRows = dims.map((d) => ({
    dim: d,
    dev: final?.dev?.pass_rate[d],
    test: final?.test?.pass_rate[d],
  }));

  // Runs where the config hash changed (or the first run): a vertical marker at
  // each on the trend charts, so a dip you see next to a marker is a config
  // change, not a regression. git_sha in the label separates a corpus/code
  // change (same config hash, different sha) from a knob change.
  const boundaries = runs
    .map((r, i) => ({ r, prev: runs[i - 1] }))
    .filter(({ r, prev }) => !prev || r.config_hash !== prev.config_hash)
    .map(({ r }) => ({
      label: `#${r.run_id}`,
      text: `${r.config_name ?? "?"} ${r.config_hash?.slice(0, 6) ?? ""} · ${r.git_sha.slice(0, 7)}`,
    }));

  const configMarkers = boundaries.map((b) => (
    <ReferenceLine
      key={b.label}
      x={b.label}
      stroke="var(--muted)"
      strokeDasharray="3 3"
      label={{ value: b.text, position: "top", fontSize: 10, fill: "var(--muted)" }}
    />
  ));

  return (
    <main className="viz-root mx-auto max-w-4xl flex-1 px-4 py-10">
      <style>{VIZ_VARS}</style>
      <h1 className="mb-1 text-2xl font-bold">Eval dashboard</h1>
      <p className="mb-8 text-sm text-gray-500">
        {runs.length} runs · latest config{" "}
        <code>{final?.config_name ?? "?"}</code> <code>{final?.config_hash?.slice(0, 8) ?? ""}</code>
      </p>

      <Section title="Pass rate per dimension" sub="Per-run judge pass rate for each rubric dimension (A–F), over time.">
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={trend} margin={{ top: 22, right: 8, left: 0, bottom: 0 }}>
            <CartesianGrid stroke="var(--grid)" vertical={false} />
            <XAxis dataKey="label" tick={{ fill: "var(--muted)", fontSize: 12 }} stroke="var(--grid)" />
            <YAxis domain={[0, 1]} tickFormatter={pct} tick={{ fill: "var(--muted)", fontSize: 12 }} stroke="var(--grid)" width={44} />
            <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v) => pct(v as number)} labelFormatter={(l, p) => `${l} · ${p?.[0]?.payload.date} · ${p?.[0]?.payload.config}`} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            {configMarkers}
            {dims.map((d) => (
              <Line key={d} dataKey={d} stroke={DIM_COLOR(d)} strokeWidth={2} dot={{ r: 3, fill: DIM_COLOR(d) }} connectNulls />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </Section>

      <Section title="Judge cost per run" sub="Total judge-call cost (USD) for each run.">
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={trend} margin={{ top: 22, right: 8, left: 0, bottom: 0 }}>
            <CartesianGrid stroke="var(--grid)" vertical={false} />
            <XAxis dataKey="label" tick={{ fill: "var(--muted)", fontSize: 12 }} stroke="var(--grid)" />
            <YAxis tickFormatter={usd} tick={{ fill: "var(--muted)", fontSize: 12 }} stroke="var(--grid)" width={64} />
            <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v) => usd(v as number)} cursor={{ fill: "var(--grid)", opacity: 0.4 }} />
            {configMarkers}
            <Bar dataKey="cost" fill="var(--seq)" radius={[4, 4, 0, 0]} maxBarSize={40} />
          </BarChart>
        </ResponsiveContainer>
      </Section>

      <Section
        title="Dev vs test — final config"
        sub={`Pass rate per dimension for the latest config, dev split vs test split${final?.dev ? ` (dev: run #${final.dev.run_id}, n=${final.dev.n}${final.test ? `; test: run #${final.test.run_id}, n=${final.test.n}` : ""})` : ""}.`}
      >
        {!final?.dev && !final?.test ? (
          <p className="text-sm text-gray-500">No judged items for the final config yet.</p>
        ) : (
          <>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={splitRows} margin={{ top: 4, right: 8, left: 0, bottom: 0 }} barGap={2}>
                <CartesianGrid stroke="var(--grid)" vertical={false} />
                <XAxis dataKey="dim" tick={{ fill: "var(--muted)", fontSize: 12 }} stroke="var(--grid)" />
                <YAxis domain={[0, 1]} tickFormatter={pct} tick={{ fill: "var(--muted)", fontSize: 12 }} stroke="var(--grid)" width={44} />
                <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v) => pct(v as number)} cursor={{ fill: "var(--grid)", opacity: 0.4 }} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Bar dataKey="dev" fill="var(--split-dev)" radius={[4, 4, 0, 0]} maxBarSize={28} />
                <Bar dataKey="test" fill="var(--split-test)" radius={[4, 4, 0, 0]} maxBarSize={28} />
              </BarChart>
            </ResponsiveContainer>
            {!final?.test && <p className="mt-2 text-sm text-gray-500">No test-split run for the final config yet.</p>}
          </>
        )}
      </Section>

      <Section title="Runs" sub="Click a run to read the judge's per-question rationales.">
        <div className="divide-y divide-gray-100 dark:divide-gray-800">
          {[...runs].reverse().map((r) => (
            <button
              key={r.run_id}
              onClick={() => setOpenRun(openRun === r.run_id ? null : r.run_id)}
              className={`flex w-full items-center gap-3 px-1 py-2 text-left text-sm hover:bg-gray-50 dark:hover:bg-gray-900 ${
                openRun === r.run_id ? "font-medium" : ""
              }`}
            >
              <span className="font-mono text-xs text-gray-400">#{r.run_id}</span>
              <span className="text-gray-500">{r.created_at.slice(0, 10)}</span>
              <span className="text-gray-400">n={r.n}</span>
              <span className="ml-auto font-mono text-xs text-gray-400">
                {dims.filter((d) => d in r.pass_rate).map((d) => `${d} ${pct(r.pass_rate[d])}`).join("  ")}
              </span>
            </button>
          ))}
        </div>
        {openRun !== null && <RunDetail key={openRun} runId={openRun} onClose={() => setOpenRun(null)} />}
      </Section>
    </main>
  );
}
