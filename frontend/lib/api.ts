import { clearToken, getToken } from "./auth";

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// Every backend request goes through here: attach the bearer token, and on a 401 drop
// the dead token and bounce to the login page.
export async function apiFetch(path: string, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers);
  const token = getToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);

  const res = await fetch(`${API_URL}${path}`, { ...init, headers });
  if (res.status === 401) {
    clearToken();
    if (typeof window !== "undefined") window.location.href = "/login";
  }
  return res;
}

// One fetch convention for JSON GETs: a non-2xx response becomes a thrown Error
// here instead of a confusing JSON-parse failure downstream.
export async function getJSON<T>(path: string): Promise<T> {
  const res = await apiFetch(path);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}
