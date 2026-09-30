// Client-side auth: the JWT lives in localStorage and rides on every backend request
// as an Authorization header (see lib/api.ts). Login/register talk to fastapi-users.
import { API_URL } from "./api";

const TOKEN_KEY = "rag_token";

export function getToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function setToken(token: string): void {
  try {
    localStorage.setItem(TOKEN_KEY, token);
  } catch {
    // Private mode / blocked storage: the session just won't persist across reloads.
  }
}

export function clearToken(): void {
  try {
    localStorage.removeItem(TOKEN_KEY);
  } catch {
    // ignore
  }
}

// fastapi-users' JWT login expects an OAuth2 form body: username (the email) + password.
export async function login(email: string, password: string): Promise<void> {
  const res = await fetch(`${API_URL}/auth/jwt/login`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({ username: email, password }),
  });
  if (!res.ok) throw new Error("Wrong email or password.");
  const { access_token } = await res.json();
  setToken(access_token);
}

export async function register(email: string, password: string, inviteCode: string): Promise<void> {
  const res = await fetch(`${API_URL}/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password, invite_code: inviteCode }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(registerError(body?.detail));
  }
}

function registerError(detail: unknown): string {
  if (detail === "INVALID_INVITE_CODE") return "That invite code isn't valid.";
  if (detail === "REGISTER_USER_ALREADY_EXISTS") return "An account with that email already exists.";
  return "Could not create the account. Check your email and password.";
}
