import type {
  AIHealthResponse,
  AISearchResponse,
  ApiErrorShape,
  ChatResponse,
  DashboardSummary,
  Report,
  ReportFormState
} from "./types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "";

function endpoint(path: string) {
  return `${API_BASE_URL}${path}`;
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

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(endpoint(path), {
      headers: {
        "Content-Type": "application/json",
        ...options?.headers
      },
      ...options
    });
  } catch (cause) {
    // Network-level failure: the server was never reached.
    throw new ApiError(0, {
      code: "NETWORK_ERROR",
      message: "Could not reach the server. Check your connection and retry."
    });
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
