// The retrieved chunk, highlighted within its document; the exact cited sentence
// is highlighted more strongly inside it.
export function Highlight({ chunk, cited }: { chunk: string; cited: string }) {
  const at = chunk.indexOf(cited);
  if (at === -1) return <mark className="bg-yellow-100">{chunk}</mark>;
  return (
    <mark className="bg-yellow-100">
      {chunk.slice(0, at)}
      <mark className="bg-yellow-300 font-medium">{cited}</mark>
      {chunk.slice(at + cited.length)}
    </mark>
  );
}
