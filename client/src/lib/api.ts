// In production, use relative URLs to proxy through Node.js server
// In development, can use direct Python backend URL
export const API_BASE = import.meta.env.VITE_API_BASE || (import.meta.env.PROD ? "" : "http://localhost:5001");

export async function apiGet<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      Accept: "application/json",
      ...(init?.headers || {}),
    },
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Request failed: ${res.status}`);
  }
  return res.json();
}


