export const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:5001";

// Warn if API_BASE is not set in production
if (import.meta.env.PROD && !import.meta.env.VITE_API_BASE) {
  console.error(
    "⚠️ VITE_API_BASE is not set! " +
    "Please set it in Vercel environment variables. " +
    "Using fallback:", API_BASE
  );
}

function checkIfHtmlResponse(text: string): boolean {
  const trimmed = text.trim();
  return trimmed.startsWith("<!DOCTYPE") || 
         trimmed.startsWith("<!doctype") || 
         trimmed.startsWith("<html") ||
         trimmed.startsWith("<HTML");
}

export async function apiDelete<T>(path: string, init?: RequestInit): Promise<T> {
  const url = `${API_BASE}${path}`;
  try {
    const res = await fetch(url, {
      ...init,
      method: "DELETE",
      headers: {
        "Content-Type": "application/json",
        ...init?.headers,
      },
    });
    const text = await res.text();

    if (!res.ok || checkIfHtmlResponse(text)) {
      if (checkIfHtmlResponse(text)) {
        throw new Error(
          `Backend returned HTML instead of JSON. This usually means:\n` +
          `1. Backend is not running or not accessible\n` +
          `2. API_BASE (${API_BASE}) is incorrect\n` +
          `3. Backend returned an error page\n\n` +
          `Please check:\n` +
          `- Is the backend running?\n` +
          `- Is VITE_API_BASE set correctly in Vercel?\n` +
          `- Check Railway logs for backend errors`
        );
      }
      throw new Error(text || `Request failed: ${res.status} ${res.statusText}`);
    }
    try {
      return JSON.parse(text) as T;
    } catch (parseError) {
      throw new Error(
        `Failed to parse JSON response. Response was: ${text.substring(0, 200)}...`
      );
    }
  } catch (error) {
    if (error instanceof TypeError && error.message.includes("fetch")) {
      throw new Error(
        `Cannot connect to backend at ${API_BASE}.\n` +
        `This usually means:\n` +
        `1. Backend is not running\n` +
        `2. CORS is not configured\n` +
        `3. Network connection issue\n\n` +
        `Please check Railway backend logs.`
      );
    }
    throw error;
  }
}
  const url = `${API_BASE}${path}`;
  
  try {
    const res = await fetch(url, {
      ...init,
      headers: {
        Accept: "application/json",
        ...(init?.headers || {}),
      },
    });
    
    // Get response text first to check if it's HTML
    const text = await res.text();
    
    // Check if response is HTML (error page)
    if (!res.ok || checkIfHtmlResponse(text)) {
      if (checkIfHtmlResponse(text)) {
        throw new Error(
          `Backend returned HTML instead of JSON. This usually means:\n` +
          `1. Backend is not running or not accessible\n` +
          `2. API_BASE (${API_BASE}) is incorrect\n` +
          `3. Backend returned an error page\n\n` +
          `Please check:\n` +
          `- Is the backend running?\n` +
          `- Is VITE_API_BASE set correctly in Vercel?\n` +
          `- Check Railway logs for backend errors`
        );
      }
      throw new Error(text || `Request failed: ${res.status} ${res.statusText}`);
    }
    
    // Parse JSON
    try {
      return JSON.parse(text) as T;
    } catch (parseError) {
      throw new Error(
        `Failed to parse JSON response. Response was: ${text.substring(0, 200)}...`
      );
    }
  } catch (error) {
    // Handle network errors
    if (error instanceof TypeError && error.message.includes("fetch")) {
      throw new Error(
        `Cannot connect to backend at ${API_BASE}.\n` +
        `This usually means:\n` +
        `1. Backend is not running\n` +
        `2. CORS is not configured\n` +
        `3. Network connection issue\n\n` +
        `Please check Railway backend logs.`
      );
    }
    throw error;
  }
}


