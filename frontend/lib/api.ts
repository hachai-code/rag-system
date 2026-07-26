export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// One fetch convention for JSON GETs: a non-2xx response becomes a thrown Error
// here instead of a confusing JSON-parse failure downstream.
export async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_URL}${path}`);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}
