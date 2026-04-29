const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:12000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
    ...init,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed with ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  base: API_BASE,
  get: request,
  post: <T>(path: string, body: unknown) =>
    request<T>(path, {
      method: "POST",
      body: JSON.stringify(body),
    }),
  fileUrl: (path: string) => `${API_BASE}/api/files/content?path=${encodeURIComponent(path)}`,
  textUrl: (path: string) => `${API_BASE}/api/files/text?path=${encodeURIComponent(path)}`,
};
