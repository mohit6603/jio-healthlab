export type ReportStatus =
  | "registered"
  | "collected"
  | "processing"
  | "review"
  | "ready"
  | "delivered";

export type ReportPriority = "routine" | "urgent";

export interface Report {
  id: number;
  patient_name: string;
  age: number;
  test_type: string;
  gender?: string | null;
  phone?: string | null;
  email?: string | null;
  city?: string | null;
  lab_branch?: string | null;
  doctor_name?: string | null;
  status: ReportStatus;
  priority: ReportPriority;
  sample_collected_at?: string | null;
  result_due_at?: string | null;
  notes?: string | null;
  created_at: string;
  updated_at: string;
}

export interface DashboardSummary {
  total_reports: number;
  ready_reports: number;
  urgent_reports: number;
  avg_age: number | null;
  unique_tests: number;
  latest_report_id: number | null;
  due_soon: Report[];
  recent_reports: Report[];
  by_status: Record<string, number>;
  by_test_type: Record<string, number>;
  by_city: Record<string, number>;
}

export interface ReportFormState {
  patient_name: string;
  age: string;
  test_type: string;
  gender: string;
  phone: string;
  email: string;
  city: string;
  lab_branch: string;
  doctor_name: string;
  status: ReportStatus;
  priority: ReportPriority;
  sample_collected_at: string;
  result_due_at: string;
  notes: string;
}

/* ── AI assistant ─────────────────────────────────────────────── */

/** A knowledge-base source an answer was grounded in. */
export interface Citation {
  title: string;
  source: string;
  chunk_id: string;
  score: number;
  section?: string | null;
  category?: string | null;
}

export interface AnswerTimings {
  retrieval_ms: number;
  generation_ms: number;
  total_ms: number;
}

/** Grounded answer returned by `POST /api/ai/chat`. */
export interface ChatResponse {
  answer: string;
  sources: Citation[];
  retrieval_count: number;
  /**
   * False when nothing relevant was retrieved, the model declined, or the
   * question crossed the clinical boundary. The UI must present these
   * differently from a grounded answer.
   */
  grounded: boolean;
  disclaimer: string;
  model: string;
  provider: string;
  finish_reason: string;
  timings: AnswerTimings;
}

export interface SearchHit {
  text: string;
  score: number;
  source: string;
  title: string;
  chunk_id: string;
  document_id?: string;
  section?: string | null;
  category?: string | null;
}

export interface AISearchResponse {
  query: string;
  results: SearchHit[];
  retrieval_count: number;
}

export interface AIComponent {
  name: string;
  state: string;
  detail?: string | null;
}

export interface AIHealthResponse {
  reachable: boolean;
  status: string;
  detail?: string | null;
  components: AIComponent[];
}

/** One turn in the assistant conversation. */
export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  text: string;
  /** Present on assistant turns that completed successfully. */
  answer?: ChatResponse;
  /** Present on assistant turns that failed. */
  error?: ApiErrorShape;
  pending?: boolean;
}

export interface ApiErrorShape {
  code: string;
  message: string;
}

/* ── AI risk analytics ────────────────────────────────────────── */

export type RiskLevel = "low" | "medium" | "high";

export interface RiskGroup {
  label: string;
  count: number;
  average_probability: number;
  high_risk: number;
}

export interface HighestRiskBranch {
  branch: string;
  average_probability: number;
  reports: number;
  high_risk: number;
}

export interface ReportRisk {
  report_id: number;
  test_type: string;
  branch?: string | null;
  city?: string | null;
  priority?: string | null;
  status?: string | null;
  result_due_at?: string | null;
  delay_probability: number;
  risk_level: RiskLevel;
}

export interface RiskAnalytics {
  generated_at: string;
  model_version: string;
  /** True when the scoring model was trained on synthetic data. */
  synthetic_model: boolean;
  reports_scored: number;
  at_risk: number;
  high_risk: number;
  predicted_late: number;
  average_probability: number;
  highest_risk_branch: HighestRiskBranch | null;
  risk_distribution: Record<string, number>;
  by_branch: RiskGroup[];
  by_test_type: RiskGroup[];
  reports: ReportRisk[];
}

/* ── Authentication ───────────────────────────────────────────── */

export type Role = "ADMIN" | "LAB_TECH" | "DOCTOR" | "VIEWER";

export interface AuthUser {
  id: number;
  email: string;
  full_name: string;
  role: Role;
  is_active: boolean;
  last_login_at?: string | null;
}

export interface TokenResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
  expires_at: string;
  user: AuthUser;
}

export interface MeResponse {
  user: AuthUser;
  permissions: string[];
}
