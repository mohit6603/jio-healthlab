import { useCallback, useEffect, useState } from "react";
import {
  AlertTriangle,
  Building2,
  FlaskConical,
  Gauge,
  Loader2,
  RefreshCw,
  ShieldAlert,
  TrendingUp
} from "lucide-react";
import { ApiError, getRiskAnalytics } from "../api";
import type { ApiErrorShape, ReportRisk, RiskAnalytics, RiskGroup } from "../types";

const RISK_TONE: Record<string, string> = {
  low: "risk-low",
  medium: "risk-medium",
  high: "risk-high"
};

function percent(value: number) {
  return `${Math.round(value * 100)}%`;
}

/**
 * Operations view of predicted delay risk.
 *
 * Deliberately shows no patient identifier: rows are keyed by report id, and
 * the API does not return names. The page is an operational triage queue, not
 * a patient listing.
 */
function AIAnalytics() {
  const [data, setData] = useState<RiskAnalytics | null>(null);
  const [error, setError] = useState<ApiErrorShape | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setData(await getRiskAnalytics(100));
    } catch (caught) {
      setError(
        caught instanceof ApiError
          ? caught.toShape()
          : { code: "UNKNOWN_ERROR", message: "Something went wrong." }
      );
      setData(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <div className="page analytics-page">
      <section className="analytics-header">
        <div>
          <p className="eyebrow">AI operations</p>
          <h1>Predicted delay risk</h1>
          <p className="analytics-intro">
            Every report still in flight, scored for the risk of missing its
            turnaround target. Queue depth and urgent load come from live data.
          </p>
        </div>
        <button
          type="button"
          className="icon-action"
          onClick={() => void load()}
          disabled={loading}
          title="Refresh"
        >
          {loading ? <Loader2 size={18} className="spin" /> : <RefreshCw size={18} />}
        </button>
      </section>

      {data?.synthetic_model && (
        <div className="analytics-notice" role="note">
          <ShieldAlert size={17} />
          <p>
            Scores come from a model trained on <strong>synthetic</strong> data.
            They demonstrate the prediction pipeline and are not an operational
            forecast.
          </p>
        </div>
      )}

      {error && <AnalyticsError error={error} onRetry={() => void load()} />}

      {loading && !data && (
        <div className="analytics-loading" role="status">
          <Loader2 size={18} className="spin" />
          <span>Scoring in-flight reports…</span>
        </div>
      )}

      {data && data.reports_scored === 0 && !error && (
        <div className="analytics-empty">
          <FlaskConical size={24} />
          <h2>Nothing in flight</h2>
          <p>
            Every report is ready or delivered, so none can miss its target.
          </p>
        </div>
      )}

      {data && data.reports_scored > 0 && (
        <>
          <section className="metrics-grid" aria-label="Risk metrics">
            <MetricTile
              icon={<AlertTriangle />}
              label="Reports at risk"
              value={data.at_risk}
              hint={`of ${data.reports_scored} scored`}
              tone="coral"
            />
            <MetricTile
              icon={<ShieldAlert />}
              label="High risk"
              value={data.high_risk}
              hint="probability ≥ 65%"
              tone="coral"
            />
            <MetricTile
              icon={<Gauge />}
              label="Average probability"
              value={percent(data.average_probability)}
              hint="across scored reports"
              tone="violet"
            />
            <MetricTile
              icon={<TrendingUp />}
              label="Predicted late"
              value={data.predicted_late}
              hint="probability ≥ 50%"
              tone="teal"
            />
            <MetricTile
              icon={<Building2 />}
              label="Highest-risk branch"
              value={data.highest_risk_branch?.branch ?? "—"}
              hint={
                data.highest_risk_branch
                  ? `${percent(data.highest_risk_branch.average_probability)} average`
                  : undefined
              }
              tone="green"
            />
          </section>

          <section className="analytics-panels">
            <RiskBreakdown
              title="Risk distribution"
              rows={[
                { label: "Low", count: data.risk_distribution.low ?? 0 },
                { label: "Medium", count: data.risk_distribution.medium ?? 0 },
                { label: "High", count: data.risk_distribution.high ?? 0 }
              ]}
              total={data.reports_scored}
            />
            <GroupPanel title="By branch" icon={<Building2 size={16} />} groups={data.by_branch} />
            <GroupPanel
              title="By test type"
              icon={<FlaskConical size={16} />}
              groups={data.by_test_type}
            />
          </section>

          <RiskTable reports={data.reports} />

          <p className="analytics-meta">
            Model {data.model_version || "unknown"} · generated{" "}
            {new Date(data.generated_at).toLocaleString()}
          </p>
        </>
      )}
    </div>
  );
}

function MetricTile({
  icon,
  label,
  value,
  hint,
  tone
}: {
  icon: React.ReactNode;
  label: string;
  value: React.ReactNode;
  hint?: string;
  tone: string;
}) {
  return (
    <div className={`metric-card ${tone}`}>
      <div className="metric-icon">{icon}</div>
      <span>{label}</span>
      <strong>{value}</strong>
      {hint && <small className="metric-hint">{hint}</small>}
    </div>
  );
}

function AnalyticsError({
  error,
  onRetry
}: {
  error: ApiErrorShape;
  onRetry: () => void;
}) {
  const guidance: Record<string, string> = {
    ML_MODEL_NOT_TRAINED:
      "No delay model has been trained yet. Run the training command, then refresh.",
    AI_SERVICE_UNAVAILABLE:
      "The AI service is offline. Reports and the dashboard still work.",
    AI_SERVICE_TIMEOUT: "Scoring took too long. Try again in a moment.",
    NETWORK_ERROR: "Could not reach the server. Check your connection and retry."
  };

  return (
    <div className="analytics-error" role="alert">
      <AlertTriangle size={18} />
      <div>
        <strong>{guidance[error.code] ?? error.message}</strong>
        <span className="assistant-error-code">{error.code}</span>
      </div>
      <button type="button" className="plain-action" onClick={onRetry}>
        Retry
      </button>
    </div>
  );
}

function RiskBreakdown({
  title,
  rows,
  total
}: {
  title: string;
  rows: { label: string; count: number }[];
  total: number;
}) {
  return (
    <div className="insight-panel">
      <div className="panel-heading compact">
        <h2>{title}</h2>
      </div>
      <ul className="bar-list">
        {rows.map((row) => (
          <li key={row.label}>
            <div className="bar-label">
              <span>{row.label}</span>
              <strong>{row.count}</strong>
            </div>
            <div className="bar-track">
              <div
                className={`bar-fill ${RISK_TONE[row.label.toLowerCase()] ?? ""}`}
                style={{ width: total ? `${(row.count / total) * 100}%` : "0%" }}
              />
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}

function GroupPanel({
  title,
  icon,
  groups
}: {
  title: string;
  icon: React.ReactNode;
  groups: RiskGroup[];
}) {
  if (groups.length === 0) {
    return (
      <div className="insight-panel">
        <div className="panel-heading compact">
          <h2>{title}</h2>
        </div>
        <div className="empty-note">Nothing to show.</div>
      </div>
    );
  }

  return (
    <div className="insight-panel">
      <div className="panel-heading compact">
        <h2>
          {icon} {title}
        </h2>
      </div>
      <ul className="bar-list">
        {groups.slice(0, 6).map((group) => (
          <li key={group.label}>
            <div className="bar-label">
              <span>{group.label}</span>
              <strong>{percent(group.average_probability)}</strong>
            </div>
            <div className="bar-track">
              <div
                className="bar-fill"
                style={{ width: `${group.average_probability * 100}%` }}
              />
            </div>
            <small className="bar-note">
              {group.count} report{group.count === 1 ? "" : "s"}
              {group.high_risk > 0 ? ` · ${group.high_risk} high risk` : ""}
            </small>
          </li>
        ))}
      </ul>
    </div>
  );
}

function RiskTable({ reports }: { reports: ReportRisk[] }) {
  return (
    <div className="reports-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Triage queue</p>
          <h2>Reports by predicted risk</h2>
        </div>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Report</th>
              <th>Test</th>
              <th>Branch</th>
              <th>Priority</th>
              <th>Status</th>
              <th>Risk</th>
              <th>Probability</th>
            </tr>
          </thead>
          <tbody>
            {reports.map((report) => (
              <tr key={report.report_id}>
                <td>#{report.report_id}</td>
                <td>{report.test_type}</td>
                <td>{report.branch ?? "—"}</td>
                <td>
                  <span className={`tag ${report.priority === "urgent" ? "tone-coral" : ""}`}>
                    {report.priority ?? "—"}
                  </span>
                </td>
                <td>{report.status ?? "—"}</td>
                <td>
                  <span className={`tag ${RISK_TONE[report.risk_level]}`}>
                    {report.risk_level}
                  </span>
                </td>
                <td className="numeric">{percent(report.delay_probability)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default AIAnalytics;
