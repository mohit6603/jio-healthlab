import type { DashboardSummary, Report, ReportFormState } from "./types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "";

function endpoint(path: string) {
  return `${API_BASE_URL}${path}`;
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(endpoint(path), {
    headers: {
      "Content-Type": "application/json",
      ...options?.headers
    },
    ...options
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed with ${response.status}`);
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
