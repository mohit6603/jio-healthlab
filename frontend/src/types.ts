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
