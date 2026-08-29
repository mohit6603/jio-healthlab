import { FormEvent, ReactNode, useEffect, useMemo, useState } from "react";
import {
  Activity,
  Bell,
  CalendarClock,
  CheckCircle2,
  ClipboardList,
  Database,
  Edit3,
  Filter,
  FlaskConical,
  Gauge,
  MapPin,
  Plus,
  RefreshCw,
  Save,
  Search,
  ShieldCheck,
  Trash2,
  UserRound
} from "lucide-react";
import labOperationsImage from "../assets/lab-operations.png";
import {
  createReport,
  deleteReport,
  getDashboard,
  getTestTypes,
  listReports,
  updateReport,
  updateReportStatus
} from "../api";
import type { DashboardSummary, Report, ReportFormState, ReportPriority, ReportStatus } from "../types";

const statusOptions: ReportStatus[] = [
  "registered",
  "collected",
  "processing",
  "review",
  "ready",
  "delivered"
];

const priorityOptions: ReportPriority[] = ["routine", "urgent"];

const defaultTests = [
  "Blood Test",
  "Sugar Test",
  "Cholesterol",
  "Thyroid",
  "CBC Panel",
  "Urine Test",
  "X-Ray",
  "MRI Scan"
];

const initialForm: ReportFormState = {
  patient_name: "",
  age: "",
  test_type: "CBC Panel",
  gender: "",
  phone: "",
  email: "",
  city: "Mumbai",
  lab_branch: "BKC Flagship",
  doctor_name: "",
  status: "registered",
  priority: "routine",
  sample_collected_at: "",
  result_due_at: "",
  notes: ""
};

const statusTone: Record<ReportStatus, string> = {
  registered: "tone-slate",
  collected: "tone-blue",
  processing: "tone-amber",
  review: "tone-violet",
  ready: "tone-green",
  delivered: "tone-graphite"
};

function Dashboard() {
  const [reports, setReports] = useState<Report[]>([]);
  const [dashboard, setDashboard] = useState<DashboardSummary | null>(null);
  const [testTypes, setTestTypes] = useState<string[]>(defaultTests);
  const [form, setForm] = useState<ReportFormState>(initialForm);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [filters, setFilters] = useState({
    search: "",
    status: "all",
    priority: "all",
    city: "all",
    test_type: "all"
  });
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    void boot();
  }, []);

  useEffect(() => {
    const timeout = window.setTimeout(() => {
      void loadReports();
    }, 180);
    return () => window.clearTimeout(timeout);
  }, [filters.search, filters.status, filters.priority, filters.city, filters.test_type]);

  async function boot() {
    setLoading(true);
    setError("");
    try {
      const [dashboardData, reportData, tests] = await Promise.all([
        getDashboard(),
        listReports(filters),
        getTestTypes()
      ]);
      setDashboard(dashboardData);
      setReports(reportData);
      setTestTypes(tests.length ? tests : defaultTests);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load Healthlab data");
    } finally {
      setLoading(false);
    }
  }

  async function loadReports() {
    try {
      const reportData = await listReports(filters);
      setReports(reportData);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load reports");
    }
  }

  async function reloadDashboard() {
    try {
      setDashboard(await getDashboard());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to refresh dashboard");
    }
  }

  async function reloadAll() {
    await Promise.all([loadReports(), reloadDashboard()]);
  }

  function updateForm<K extends keyof ReportFormState>(field: K, value: ReportFormState[K]) {
    setForm((current) => ({ ...current, [field]: value }));
  }

  function editReport(report: Report) {
    setEditingId(report.id);
    setForm(reportToForm(report));
    document.querySelector(".editor-panel")?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function resetForm() {
    setEditingId(null);
    setForm(initialForm);
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSaving(true);
    setError("");
    try {
      if (editingId) {
        await updateReport(editingId, form);
      } else {
        await createReport(form);
      }
      resetForm();
      await reloadAll();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to save report");
    } finally {
      setSaving(false);
    }
  }

  async function handleDelete(id: number) {
    const confirmed = window.confirm(`Delete report #${id}?`);
    if (!confirmed) return;
    setError("");
    try {
      await deleteReport(id);
      await reloadAll();
      if (editingId === id) resetForm();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to delete report");
    }
  }

  async function markDelivered(report: Report) {
    setError("");
    try {
      await updateReportStatus(report.id, "delivered");
      await reloadAll();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to update status");
    }
  }

  const cities = useMemo(() => {
    const source = dashboard?.by_city ?? {};
    return Object.keys(source).filter((city) => city !== "Unknown").sort();
  }, [dashboard]);

  const topTest = useMemo(() => topLabel(dashboard?.by_test_type), [dashboard]);
  const statusChart = useMemo(() => toChartItems(dashboard?.by_status), [dashboard]);
  const testChart = useMemo(() => toChartItems(dashboard?.by_test_type), [dashboard]);

  return (
    <div className="page">
      <section className="command-banner" style={{ backgroundImage: `url(${labOperationsImage})` }}>
        <div className="banner-content">
          <div className="brand-mark">
            <FlaskConical size={22} />
            <span>JIO Healthlab</span>
          </div>
          <h1>Live diagnostic operations</h1>
          <p>
            Track report intake, sample progress, urgent queues, branch load, and delivery readiness from one control surface.
          </p>
          <div className="banner-actions">
            <button className="primary-action" type="button" onClick={resetForm}>
              <Plus size={18} />
              New report
            </button>
            <a className="secondary-action" href="/docs" target="_blank" rel="noreferrer">
              <Database size={18} />
              API docs
            </a>
          </div>
        </div>
        <div className="banner-strip">
          <span>Top test: {topTest}</span>
          <span>Latest report: {dashboard?.latest_report_id ? `#${dashboard.latest_report_id}` : "None"}</span>
          <span>Queue: {reports.length} visible</span>
        </div>
      </section>

      {error && <div className="error-banner">{error}</div>}

      <section className="metrics-grid" aria-label="Lab metrics">
        <MetricCard icon={<ClipboardList />} label="Total reports" value={dashboard?.total_reports ?? 0} tone="teal" />
        <MetricCard icon={<CheckCircle2 />} label="Ready" value={dashboard?.ready_reports ?? 0} tone="green" />
        <MetricCard icon={<Bell />} label="Urgent" value={dashboard?.urgent_reports ?? 0} tone="coral" />
        <MetricCard icon={<Gauge />} label="Avg age" value={dashboard?.avg_age ? `${dashboard.avg_age} yrs` : "--"} tone="violet" />
      </section>

      <section className="workspace-grid">
        <div className="reports-panel">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">Report registry</p>
              <h2>Patient reports</h2>
            </div>
            <button className="icon-action" type="button" onClick={() => void reloadAll()} title="Refresh">
              <RefreshCw size={18} />
            </button>
          </div>

          <div className="filters-bar">
            <label className="search-field">
              <Search size={18} />
              <input
                value={filters.search}
                onChange={(event) => setFilters({ ...filters, search: event.target.value })}
                placeholder="Search patient, doctor, branch, ID"
              />
            </label>
            <FilterSelect
              label="Status"
              value={filters.status}
              onChange={(value) => setFilters({ ...filters, status: value })}
              options={["all", ...statusOptions]}
            />
            <FilterSelect
              label="Priority"
              value={filters.priority}
              onChange={(value) => setFilters({ ...filters, priority: value })}
              options={["all", ...priorityOptions]}
            />
            <FilterSelect
              label="City"
              value={filters.city}
              onChange={(value) => setFilters({ ...filters, city: value })}
              options={["all", ...cities]}
            />
          </div>

          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Patient</th>
                  <th>Test</th>
                  <th>Branch</th>
                  <th>Status</th>
                  <th>Due</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {loading ? (
                  <tr>
                    <td colSpan={7} className="empty-cell">Loading reports...</td>
                  </tr>
                ) : reports.length ? (
                  reports.map((report) => (
                    <tr key={report.id}>
                      <td>
                        <span className="id-pill">#{report.id}</span>
                      </td>
                      <td>
                        <div className="patient-cell">
                          <UserRound size={16} />
                          <div>
                            <strong>{report.patient_name}</strong>
                            <span>{report.age} yrs {report.gender ? `- ${report.gender}` : ""}</span>
                          </div>
                        </div>
                      </td>
                      <td>
                        <strong>{report.test_type}</strong>
                        <span className={`priority ${report.priority}`}>{report.priority}</span>
                      </td>
                      <td>
                        <span>{report.lab_branch ?? "Unassigned"}</span>
                        <small>{report.city ?? "Unknown city"}</small>
                      </td>
                      <td>
                        <span className={`status-pill ${statusTone[report.status]}`}>{labelize(report.status)}</span>
                      </td>
                      <td>{formatDateTime(report.result_due_at)}</td>
                      <td>
                        <div className="row-actions">
                          <button className="icon-action" type="button" onClick={() => editReport(report)} title="Edit">
                            <Edit3 size={16} />
                          </button>
                          <button className="icon-action" type="button" onClick={() => void markDelivered(report)} title="Mark delivered">
                            <ShieldCheck size={16} />
                          </button>
                          <button className="icon-action danger" type="button" onClick={() => void handleDelete(report.id)} title="Delete">
                            <Trash2 size={16} />
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={7} className="empty-cell">No reports match the current filters.</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        <aside className="editor-panel">
          <div className="panel-heading compact">
            <div>
              <p className="eyebrow">{editingId ? `Editing #${editingId}` : "Create report"}</p>
              <h2>{editingId ? "Update report" : "New intake"}</h2>
            </div>
            {editingId && (
              <button className="icon-action" type="button" onClick={resetForm} title="Clear">
                <Plus size={18} />
              </button>
            )}
          </div>

          <form onSubmit={handleSubmit} className="report-form">
            <Field label="Patient name">
              <input required value={form.patient_name} onChange={(event) => updateForm("patient_name", event.target.value)} />
            </Field>
            <div className="form-row">
              <Field label="Age">
                <input required type="number" min="0" max="120" value={form.age} onChange={(event) => updateForm("age", event.target.value)} />
              </Field>
              <Field label="Gender">
                <input value={form.gender} onChange={(event) => updateForm("gender", event.target.value)} />
              </Field>
            </div>
            <Field label="Test type">
              <select required value={form.test_type} onChange={(event) => updateForm("test_type", event.target.value)}>
                {testTypes.map((test) => (
                  <option key={test} value={test}>{test}</option>
                ))}
              </select>
            </Field>
            <div className="form-row">
              <Field label="Status">
                <select value={form.status} onChange={(event) => updateForm("status", event.target.value as ReportStatus)}>
                  {statusOptions.map((status) => (
                    <option key={status} value={status}>{labelize(status)}</option>
                  ))}
                </select>
              </Field>
              <Field label="Priority">
                <select value={form.priority} onChange={(event) => updateForm("priority", event.target.value as ReportPriority)}>
                  {priorityOptions.map((priority) => (
                    <option key={priority} value={priority}>{labelize(priority)}</option>
                  ))}
                </select>
              </Field>
            </div>
            <div className="form-row">
              <Field label="City">
                <input value={form.city} onChange={(event) => updateForm("city", event.target.value)} />
              </Field>
              <Field label="Branch">
                <input value={form.lab_branch} onChange={(event) => updateForm("lab_branch", event.target.value)} />
              </Field>
            </div>
            <Field label="Doctor">
              <input value={form.doctor_name} onChange={(event) => updateForm("doctor_name", event.target.value)} />
            </Field>
            <div className="form-row">
              <Field label="Phone">
                <input value={form.phone} onChange={(event) => updateForm("phone", event.target.value)} />
              </Field>
              <Field label="Email">
                <input type="email" value={form.email} onChange={(event) => updateForm("email", event.target.value)} />
              </Field>
            </div>
            <div className="form-row">
              <Field label="Collected">
                <input type="datetime-local" value={form.sample_collected_at} onChange={(event) => updateForm("sample_collected_at", event.target.value)} />
              </Field>
              <Field label="Result due">
                <input type="datetime-local" value={form.result_due_at} onChange={(event) => updateForm("result_due_at", event.target.value)} />
              </Field>
            </div>
            <Field label="Notes">
              <textarea rows={3} value={form.notes} onChange={(event) => updateForm("notes", event.target.value)} />
            </Field>
            <div className="form-actions">
              <button className="save-action" type="submit" disabled={saving}>
                <Save size={18} />
                {saving ? "Saving" : editingId ? "Update report" : "Create report"}
              </button>
              <button className="plain-action" type="button" onClick={resetForm}>Clear</button>
            </div>
          </form>
        </aside>
      </section>

      <section className="insights-grid">
        <InsightPanel title="Status mix" icon={<Activity size={18} />} items={statusChart} />
        <InsightPanel title="Test demand" icon={<FlaskConical size={18} />} items={testChart} />
        <div className="queue-panel">
          <div className="panel-heading compact">
            <div>
              <p className="eyebrow">Next 24 hours</p>
              <h2>Due soon</h2>
            </div>
            <CalendarClock size={20} />
          </div>
          <div className="queue-list">
            {dashboard?.due_soon.length ? (
              dashboard.due_soon.map((report) => (
                <button key={report.id} type="button" className="queue-item" onClick={() => editReport(report)}>
                  <span className="id-pill">#{report.id}</span>
                  <strong>{report.patient_name}</strong>
                  <small>{formatDateTime(report.result_due_at)}</small>
                </button>
              ))
            ) : (
              <div className="empty-note">No urgent due-soon reports.</div>
            )}
          </div>
        </div>
      </section>
    </div>
  );
}

function MetricCard({ icon, label, value, tone }: { icon: ReactNode; label: string; value: ReactNode; tone: string }) {
  return (
    <div className={`metric-card ${tone}`}>
      <div className="metric-icon">{icon}</div>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
    </label>
  );
}

function FilterSelect({
  label,
  value,
  options,
  onChange
}: {
  label: string;
  value: string;
  options: string[];
  onChange: (value: string) => void;
}) {
  return (
    <label className="filter-select">
      <Filter size={15} />
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {options.map((option) => (
          <option key={option} value={option}>{option === "all" ? "All" : labelize(option)}</option>
        ))}
      </select>
    </label>
  );
}

function InsightPanel({ title, icon, items }: { title: string; icon: ReactNode; items: ChartItem[] }) {
  return (
    <div className="insight-panel">
      <div className="panel-heading compact">
        <div>
          <p className="eyebrow">Live view</p>
          <h2>{title}</h2>
        </div>
        {icon}
      </div>
      <div className="bar-list">
        {items.length ? (
          items.map((item) => (
            <div className="bar-row" key={item.label}>
              <div className="bar-meta">
                <span>{labelize(item.label)}</span>
                <strong>{item.value}</strong>
              </div>
              <div className="bar-track">
                <div className="bar-fill" style={{ width: `${item.percent}%` }} />
              </div>
            </div>
          ))
        ) : (
          <div className="empty-note">No data yet.</div>
        )}
      </div>
    </div>
  );
}

interface ChartItem {
  label: string;
  value: number;
  percent: number;
}

function toChartItems(values?: Record<string, number>): ChartItem[] {
  const entries = Object.entries(values ?? {}).sort((a, b) => b[1] - a[1]).slice(0, 6);
  const max = Math.max(...entries.map(([, value]) => value), 1);
  return entries.map(([label, value]) => ({
    label,
    value,
    percent: Math.max(8, Math.round((value / max) * 100))
  }));
}

function topLabel(values?: Record<string, number>) {
  const [first] = Object.entries(values ?? {}).sort((a, b) => b[1] - a[1]);
  return first?.[0] ?? "None";
}

function reportToForm(report: Report): ReportFormState {
  return {
    patient_name: report.patient_name,
    age: String(report.age),
    test_type: report.test_type,
    gender: report.gender ?? "",
    phone: report.phone ?? "",
    email: report.email ?? "",
    city: report.city ?? "",
    lab_branch: report.lab_branch ?? "",
    doctor_name: report.doctor_name ?? "",
    status: report.status,
    priority: report.priority,
    sample_collected_at: toDateTimeInput(report.sample_collected_at),
    result_due_at: toDateTimeInput(report.result_due_at),
    notes: report.notes ?? ""
  };
}

function toDateTimeInput(value?: string | null) {
  if (!value) return "";
  const date = new Date(value);
  const local = new Date(date.getTime() - date.getTimezoneOffset() * 60000);
  return local.toISOString().slice(0, 16);
}

function formatDateTime(value?: string | null) {
  if (!value) return "Not set";
  return new Intl.DateTimeFormat("en-IN", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit"
  }).format(new Date(value));
}

function labelize(value: string) {
  return value.replace(/_/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export default Dashboard;
