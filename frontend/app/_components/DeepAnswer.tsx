import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { CorpusSource } from "@/lib/types";
import { CorpusSources } from "./CorpusSources";
import { WebSources } from "./WebSources";

// A deep-agent answer rendered in full: Markdown body (headings, quotes, tables),
// then the corpus-source chips and cited web links. Shared by the live chat view
// and the memory viewer so the two can't drift.
export function DeepAnswer({
  answer,
  corpusSources,
}: {
  answer: string;
  corpusSources: CorpusSource[];
}) {
  return (
    <>
      <article className="prose prose-neutral mb-8 max-w-none">
        <Markdown remarkPlugins={[remarkGfm]}>{answer}</Markdown>
      </article>
      {corpusSources.length > 0 && <CorpusSources sources={corpusSources} />}
      <WebSources answer={answer} />
    </>
  );
}
