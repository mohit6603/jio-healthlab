import type {
  AIHealthResponse,
  AISearchResponse,
  ApiErrorShape,
  ChatResponse,
  Citation,
  DashboardSummary,
  MeResponse,
  Report,
  ReportFormState,
  RiskAnalytics,
  TokenResponse
} from "./types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "";

function endpoint(path: string) {
  return `${API_BASE_URL}${path}`;
}

/* ── Token handling ───────────────────────────────────────────────
 *
 * The access token is held in memory only. The refresh token lives in
 * localStorage so a page reload can restore the session silently.
 *
 * This is a deliberate compromise. Neither store survives an XSS attack, but
 * keeping the short-lived access token out of persistent storage limits what
 * a successful injection can exfiltrate, and the refresh token is revocable
 * server-side and rotated on every use. httpOnly, SameSite cookies would be
 * the stronger choice and are noted in docs/security.md.
 */

const REFRESH_STORAGE_KEY = "healthlab.refresh_token";

let accessToken: string | null = null;
let onSessionLost: (() => void) | null = null;

export function setAccessToken(token: string | null) {
  accessToken = token;
}

export function getAccessToken() {
  return accessToken;
}

export function setRefreshToken(token: string | null) {
  try {
    if (token) window.localStorage.setItem(REFRESH_STORAGE_KEY, token);
    else window.localStorage.removeItem(REFRESH_STORAGE_KEY);
  } catch {
    // Private browsing or blocked storage: the session simply will not
    // survive a reload.
  }
}

export function getRefreshToken(): string | null {
  try {
    return window.localStorage.getItem(REFRESH_STORAGE_KEY);
  } catch {
    return null;
  }
}

/** Called when the session cannot be recovered, so the app can sign out. */
export function setSessionLostHandler(handler: (() => void) | null) {
  onSessionLost = handler;
}

/**
 * Error carrying the API's `{"error": {code, message}}` envelope.
 *
 * The code matters to the UI: `AI_SERVICE_UNAVAILABLE` and
 * `GENERATION_DISABLED` need different messages, and only the server knows
 * which applies.
 */
export class ApiError extends Error {
  readonly code: string;
  readonly status: number;

  constructor(status: number, body: ApiErrorShape) {
    super(body.message);
    this.name = "ApiError";
    this.code = body.code;
    this.status = status;
  }

  toShape(): ApiErrorShape {
    return { code: this.code, message: this.message };
  }
}

async function parseError(response: Response): Promise<ApiError> {
  try {
    const body = await response.json();
    if (body && typeof body === "object" && "error" in body) {
      const envelope = (body as { error: ApiErrorShape }).error;
      return new ApiError(response.status, envelope);
    }
  } catch {
    // Fall through to a generic message below.
  }
  return new ApiError(response.status, {
    code: "REQUEST_FAILED",
    message: `Request failed with status ${response.status}.`
  });
}

async function send(path: string, options?: RequestInit): Promise<Response> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...((options?.headers as Record<string, string>) ?? {})
  };
  if (accessToken) headers.Authorization = `Bearer ${accessToken}`;

  try {
    return await fetch(endpoint(path), { ...options, headers });
  } catch {
    // Network-level failure: the server was never reached.
    throw new ApiError(0, {
      code: "NETWORK_ERROR",
      message: "Could not reach the server. Check your connection and retry."
    });
  }
}

/** Single in-flight refresh, so a burst of 401s does not rotate N times. */
let refreshInFlight: Promise<boolean> | null = null;

async function refreshSession(): Promise<boolean> {
  const token = getRefreshToken();
  if (!token) return false;

  if (!refreshInFlight) {
    refreshInFlight = (async () => {
      try {
        const response = await fetch(endpoint("/api/auth/refresh"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ refresh_token: token })
        });
        if (!response.ok) return false;
        const session = (await response.json()) as {
          access_token: string;
          refresh_token: string;
        };
        setAccessToken(session.access_token);
        setRefreshToken(session.refresh_token);
        return true;
      } catch {
        return false;
      } finally {
        refreshInFlight = null;
      }
    })();
  }

  return refreshInFlight;
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  let response = await send(path, options);

  // One transparent retry: the access token is short-lived by design, so an
  // expired token during normal use is expected rather than exceptional.
  if (response.status === 401 && !path.startsWith("/api/auth/")) {
    if (await refreshSession()) {
      response = await send(path, options);
    } else {
      setAccessToken(null);
      setRefreshToken(null);
      onSessionLost?.();
    }
  }

  if (!response.ok) {
    throw await parseError(response);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return response.json() as Promise<T>;
}

export interface ReportFilters {
  search?: string;
  status?: string;
  priority?: string;
  city?: string;
  test_type?: string;
}

export function listReports(filters: ReportFilters = {}) {
  const params = new URLSearchParams();
  Object.entries(filters).forEach(([key, value]) => {
    if (value && value !== "all") params.set(key, value);
  });
  const suffix = params.toString() ? `?${params.toString()}` : "";
  return request<Report[]>(`/api/reports${suffix}`);
}

export function getDashboard() {
  return request<DashboardSummary>("/api/dashboard");
}

export function getTestTypes() {
  return request<string[]>("/api/test-types");
}

function formToPayload(form: ReportFormState) {
  return {
    patient_name: form.patient_name,
    age: Number(form.age),
    test_type: form.test_type,
    gender: emptyToNull(form.gender),
    phone: emptyToNull(form.phone),
    email: emptyToNull(form.email),
    city: emptyToNull(form.city),
    lab_branch: emptyToNull(form.lab_branch),
    doctor_name: emptyToNull(form.doctor_name),
    status: form.status,
    priority: form.priority,
    sample_collected_at: dateTimeToIso(form.sample_collected_at),
    result_due_at: dateTimeToIso(form.result_due_at),
    notes: emptyToNull(form.notes)
  };
}

function emptyToNull(value: string) {
  return value.trim() ? value.trim() : null;
}

function dateTimeToIso(value: string) {
  return value ? new Date(value).toISOString() : null;
}

export function createReport(form: ReportFormState) {
  return request<Report>("/api/reports", {
    method: "POST",
    body: JSON.stringify(formToPayload(form))
  });
}

export function updateReport(id: number, form: ReportFormState) {
  return request<Report>(`/api/reports/${id}`, {
    method: "PUT",
    body: JSON.stringify(formToPayload(form))
  });
}

export function updateReportStatus(id: number, status: Report["status"]) {
  return request<Report>(`/api/reports/${id}`, {
    method: "PUT",
    body: JSON.stringify({ status })
  });
}

export function deleteReport(id: number) {
  return request<void>(`/api/reports/${id}`, {
    method: "DELETE"
  });
}

/* ── AI assistant ─────────────────────────────────────────────── */

export interface ChatOptions {
  topK?: number;
  category?: string;
}

/** Ask the knowledge assistant a grounded question. */
export function askAssistant(question: string, options: ChatOptions = {}) {
  return request<ChatResponse>("/api/ai/chat", {
    method: "POST",
    body: JSON.stringify({
      question,
      top_k: options.topK ?? null,
      category: options.category ?? null
    })
  });
}

/** Semantic search with no generation -- works when the LLM is unavailable. */
export function searchKnowledge(query: string, options: ChatOptions = {}) {
  return request<AISearchResponse>("/api/ai/search", {
    method: "POST",
    body: JSON.stringify({ query, top_k: options.topK ?? null })
  });
}

/** AI service availability. Never rejects for a down dependency. */
export function getAIHealth() {
  return request<AIHealthResponse>("/api/ai/health");
}

/** Predicted delay risk across in-flight reports. */
export function getRiskAnalytics(limit = 100) {
  return request<RiskAnalytics>(`/api/ai/risk-analytics?limit=${limit}`);
}

/* ── Authentication ───────────────────────────────────────────── */

export function login(email: string, password: string) {
  return request<TokenResponse>("/api/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password })
  });
}

export function refreshWithToken(refreshToken: string) {
  return request<TokenResponse>("/api/auth/refresh", {
    method: "POST",
    body: JSON.stringify({ refresh_token: refreshToken })
  });
}

export function getMe() {
  return request<MeResponse>("/api/auth/me");
}

export function logout(refreshToken: string | null) {
  return request<{ sessions_ended: number }>("/api/auth/logout", {
    method: "POST",
    body: JSON.stringify({ refresh_token: refreshToken })
  });
}

/* ── Streaming assistant ──────────────────────────────────────── */

export interface StreamHandlers {
  onSources?: (sources: Citation[]) => void;
  onToken?: (text: string) => void;
  /** The safety screen rejected the finished answer; discard what was shown. */
  onReplace?: (text: string) => void;
  onDone?: (summary: StreamSummary) => void;
  onError?: (error: ApiErrorShape) => void;
}

export interface StreamSummary {
  grounded: boolean;
  finish_reason: string;
  model: string;
  provider: string;
  retrieval_count: number;
  drop_sources: boolean;
  disclaimer: string;
  timings: { retrieval_ms: number; generation_ms: number; total_ms: number };
}

/**
 * Ask the assistant over Server-Sent Events.
 *
 * `fetch` with a reader rather than `EventSource`, because EventSource cannot
 * send a POST body or an Authorization header. Returns an abort function so a
 * component can cancel on unmount.
 */
export function streamAssistant(
  question: string,
  handlers: StreamHandlers,
  options: ChatOptions = {}
): () => void {
  const controller = new AbortController();

  void (async () => {
    let response: Response;
    try {
      response = await send("/api/ai/chat/stream", {
        method: "POST",
        body: JSON.stringify({
          question,
          top_k: options.topK ?? null,
          category: options.category ?? null
        }),
        headers: { Accept: "text/event-stream" },
        signal: controller.signal
      });
    } catch (caught) {
      if (!controller.signal.aborted) {
        handlers.onError?.(
          caught instanceof ApiError
            ? caught.toShape()
            : { code: "NETWORK_ERROR", message: "Could not reach the server." }
        );
      }
      return;
    }

    // A 401 here cannot be retried transparently mid-stream, so surface it.
    if (!response.ok || !response.body) {
      handlers.onError?.((await parseError(response)).toShape());
      return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // Frames are separated by a blank line; a partial frame stays in the
        // buffer until the rest arrives.
        let split: number;
        while ((split = buffer.indexOf("\n\n")) !== -1) {
          const frame = buffer.slice(0, split);
          buffer = buffer.slice(split + 2);
          dispatch(frame, handlers);
        }
      }
    } catch {
      if (!controller.signal.aborted) {
        handlers.onError?.({
          code: "STREAM_INTERRUPTED",
          message: "The connection dropped before the answer finished."
        });
      }
    }
  })();

  return () => controller.abort();
}

function dispatch(frame: string, handlers: StreamHandlers) {
  let event = "message";
  const dataLines: string[] = [];

  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }
  if (dataLines.length === 0) return;

  let payload: unknown;
  try {
    payload = JSON.parse(dataLines.join("\n"));
  } catch {
    return; // A malformed frame is skipped rather than killing the stream.
  }

  switch (event) {
    case "sources":
      handlers.onSources?.((payload as { sources: Citation[] }).sources);
      break;
    case "token":
      handlers.onToken?.((payload as { text: string }).text);
      break;
    case "replace":
      handlers.onReplace?.((payload as { text: string }).text);
      break;
    case "done":
      handlers.onDone?.(payload as StreamSummary);
      break;
    case "error":
      handlers.onError?.(payload as ApiErrorShape);
      break;
  }
}
